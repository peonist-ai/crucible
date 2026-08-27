"""Tests for `crucible plan`: type facts, budget solving, pin resolution, tier presets."""

from __future__ import annotations

import struct

import pytest

from crucible.allocate import (
    Group,
    anchored,
    plan,
    regroup_by_role,
    render_tensor_type_file,
    role_pattern,
)
from crucible.gguf import Tensor, read_header
from crucible.quant_types import QUANT_TYPES, REQUIRES_IMATRIX, ladder
from crucible.tiers import TIERS, parse_size

GIB = 1024**3

# Measured off the shipped Qwen3.6-35B-A3B-REAP-48-Q3K-mixed GGUF (Qwen 3.6 35B-A3B
# REAP-48, 40 layers, 133 experts). (role, n_tensors, total_params, shipped_type).
# The file is 8.770 GiB on disk; this inventory must reproduce that from the
# block layouts alone, which is what makes every projected size trustworthy.
REAP48 = [
    ("ffn_down_exps.weight",  40, 5_578_424_320, "Q3_K"),
    ("ffn_gate_exps.weight",  40, 5_578_424_320, "Q3_K"),
    ("ffn_up_exps.weight",    40, 5_578_424_320, "Q3_K"),
    ("output.weight",          1,   508_559_360, "Q8_0"),
    ("token_embd.weight",      1,   508_559_360, "Q8_0"),
    ("attn_qkv.weight",       30,   503_316_480, "Q8_0"),
    ("attn_q.weight",         10,   167_772_160, "Q8_0"),
    ("attn_gate.weight",      30,   251_658_240, "Q3_K"),
    ("ssm_out.weight",        30,   251_658_240, "Q3_K"),
    ("attn_output.weight",    10,    83_886_080, "Q8_0"),
    ("ffn_down_shexp.weight", 40,    41_943_040, "Q3_K"),
    ("ffn_gate_shexp.weight", 40,    41_943_040, "Q3_K"),
    ("ffn_up_shexp.weight",   40,    41_943_040, "Q3_K"),
    ("attn_k.weight",         10,    10_485_760, "Q8_0"),
    ("attn_v.weight",         10,    10_485_760, "Q8_0"),
    ("ssm_alpha.weight",      30,     1_966_080, "Q3_K"),
    ("ssm_beta.weight",       30,     1_966_080, "Q3_K"),
]
REAP48_F32 = [("ffn_gate_inp.weight", 40, 10_895_360), ("ssm_conv1d.weight", 30, 983_040)]


def reap48_groups() -> list[Group]:
    gs = [Group(r, role_pattern(r), n, n_tensors=k) for r, k, n, _ in REAP48]
    gs += [Group(r, role_pattern(r), n, n_tensors=k, quantizable=False)
           for r, k, n in REAP48_F32]
    return gs


class TestQuantTypes:
    def test_bpw_is_exact_block_arithmetic(self):
        # These are the block layouts in ggml-common.h, not the round file sizes
        # printed by `llama-quantize --help`.
        assert QUANT_TYPES["Q3_K"].bpw == 3.4375
        assert QUANT_TYPES["Q4_K"].bpw == 4.5
        assert QUANT_TYPES["Q8_0"].bpw == 8.5
        assert QUANT_TYPES["IQ4_XS"].bpw == 4.25
        # Q2_K is 2.625 bpw. The help text's "2.96G" is the whole-file size of
        # the Q2_K *mixture*, which holds attention far above 2 bits.
        assert QUANT_TYPES["Q2_K"].bpw == 2.625

    def test_iq3s_is_free_against_q3k(self):
        """Identical size, lower measured error — the swap costs nothing."""
        iq3s, q3k = QUANT_TYPES["IQ3_S"], QUANT_TYPES["Q3_K"]
        assert iq3s.bpw == q3k.bpw
        assert iq3s.rmse < q3k.rmse

    def test_ladder_ascends_in_size(self):
        rungs = ladder()
        for lo, hi in zip(rungs, rungs[1:]):
            assert lo.bpw < hi.bpw

    def test_ladder_is_not_assumed_monotone_in_quality(self):
        """IQ2_XXS measures worse than the smaller IQ1_M, and that is expected.

        It is one of the types ggml refuses to produce without an imatrix, and
        the distortion column was measured with a flat one — so the number is an
        artefact of the measurement, not a property of the format. Pinned here so
        that nobody "corrects" the table by hand, and so the invariant the
        planner actually relies on is the hull's, not the ladder's.
        """
        assert QUANT_TYPES["IQ2_XXS"].bpw > QUANT_TYPES["IQ1_M"].bpw
        assert QUANT_TYPES["IQ2_XXS"].rmse > QUANT_TYPES["IQ1_M"].rmse
        assert "IQ2_XXS" in REQUIRES_IMATRIX

    def test_hull_is_monotone_even_though_the_ladder_is_not(self):
        """What the greedy actually needs: strictly better as it gets bigger."""
        from crucible.allocate import _hull

        rungs = _hull(ladder(), Group("g", role_pattern("g"), 10_000_000))
        for lo, hi in zip(rungs, rungs[1:]):
            assert lo.bpw < hi.bpw
            assert lo.rmse > hi.rmse, f"{lo.name} -> {hi.name} survived the hull"

    def test_ladder_drops_imatrix_only_types_when_blind(self):
        blind = {q.name for q in ladder(have_imatrix=False)}
        assert not (blind & REQUIRES_IMATRIX)
        assert REQUIRES_IMATRIX & {q.name for q in ladder(have_imatrix=True)}

    def test_ladder_excludes_backend_hostile_types(self):
        assert "Q2_K" not in {q.name for q in ladder(exclude=("Q2_K",))}

    def test_ladder_raises_rather_than_returning_nothing(self):
        with pytest.raises(ValueError, match="no admissible"):
            ladder(floor="Q8_0", ceiling="Q2_K")

    def test_bytes_round_up_to_whole_blocks(self):
        q = QUANT_TYPES["Q4_K"]
        assert q.bytes_for(256) == 144
        assert q.bytes_for(257) == 288  # a second block, however little is in it


class TestPlanner:
    def test_reproduces_the_shipped_file_size(self):
        """The inventory plus block arithmetic must land on the real 8.770 GiB."""
        shipped = {r: t for r, _, _, t in REAP48}
        total = sum(g.bytes_for(QUANT_TYPES[shipped[g.key]])
                    for g in reap48_groups() if g.quantizable)
        total += sum(g.bytes_for(QUANT_TYPES["F32"])
                     for g in reap48_groups() if not g.quantizable)
        assert total / GIB == pytest.approx(8.770, abs=0.005)

    def test_never_exceeds_the_budget(self):
        for budget_gib in (7.0, 8.769, 11.0, 16.0):
            result = plan(reap48_groups(), ladder(), int(budget_gib * GIB))
            assert result.total_bytes <= int(budget_gib * GIB)
            assert result.fits

    def test_spends_the_budget_it_is_given(self):
        """A capacity tier wastes nothing: unspent bytes are quality left behind."""
        budget = int(11.0 * GIB)
        result = plan(reap48_groups(), ladder(), budget)
        assert result.total_bytes > budget * 0.98

    def test_importance_moves_bits(self):
        """Two identical groups differing only in activation energy diverge."""
        gs = [
            Group("quiet", role_pattern("quiet"), 100_000_000, energy=1.0),
            Group("loud", role_pattern("loud"), 100_000_000, energy=100.0),
        ]
        result = plan(gs, ladder(), 120_000_000)
        chosen = {c.group.key: c.quant.bpw for c in result.choices}
        assert chosen["loud"] > chosen["quiet"]

    def test_pins_are_honoured(self):
        result = plan(reap48_groups(), ladder(), int(11.0 * GIB),
                      pins={"ffn_down_shexp.weight": "Q8_0"})
        chosen = {c.group.key: c.quant.name for c in result.choices}
        assert chosen["ffn_down_shexp.weight"] == "Q8_0"

    def test_a_pin_reaches_every_layer_of_a_role(self):
        """Per-layer groups key on tensor name; a role-keyed pin must still land."""
        gs = [
            Group(f"blk.{i}.attn_gate.weight", anchored(f"blk.{i}.attn_gate.weight"),
                  8_388_608, role="attn_gate.weight")
            for i in range(30)
        ]
        result = plan(gs, ladder(), 10 * GIB, pins={"attn_gate.weight": "Q6_K"})
        assert len(result.choices) == 30
        assert all(c.quant.name == "Q6_K" for c in result.choices)

    def test_a_pin_matching_nothing_raises(self):
        """Silently dropping it would make the plan lie about what it applied."""
        with pytest.raises(ValueError, match="matched no tensor"):
            plan(reap48_groups(), ladder(), int(9 * GIB), pins={"ffn_typo.weight": "Q8_0"})

    def test_key_pin_beats_role_pin(self):
        gs = [Group("blk.0.attn_gate.weight", anchored("blk.0.attn_gate.weight"),
                    8_388_608, role="attn_gate.weight")]
        result = plan(gs, ladder(), 10 * GIB,
                      pins={"attn_gate.weight": "Q6_K", "blk.0.attn_gate.weight": "Q8_0"})
        assert result.choices[0].quant.name == "Q8_0"

    def test_an_impossible_pin_reports_rather_than_substitutes(self):
        """Silently swapping in something cheaper is how a budget is missed by 400MB."""
        result = plan(reap48_groups(), ladder(), int(4.0 * GIB),
                      pins={r: "Q8_0" for r, _, _, _ in REAP48})
        assert not result.fits
        assert all(c.quant.name == "Q8_0" for c in result.choices)

    def test_unquantizable_groups_are_counted_as_f32(self):
        result = plan(reap48_groups(), ladder(), int(11.0 * GIB))
        assert result.fixed_bytes == sum(n * 4 for _, _, n in REAP48_F32)
        assert all(c.group.quantizable for c in result.choices)

    def test_dominated_rungs_are_never_chosen(self):
        """Q4_0 and Q4_K are both 4.5 bpw; only one of them is ever right."""
        gs = [Group("x", role_pattern("x"), 100_000_000)]
        rungs = tuple(QUANT_TYPES[n] for n in ("Q3_K", "Q4_0", "Q4_K", "Q5_K"))
        result = plan(gs, rungs, 60_000_000)
        assert result.choices[0].quant.name != "Q4_0"

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="empty quantization ladder"):
            plan(reap48_groups(), (), int(8 * GIB))

    def test_unknown_pin_type_raises(self):
        with pytest.raises(ValueError, match="unknown quantization type"):
            plan(reap48_groups(), ladder(), int(8 * GIB), pins={"output.weight": "Q9_Z"})


class TestPatterns:
    def test_role_pattern_does_not_overmatch(self):
        """llama.cpp uses regex_search, so a bare `ffn_gate` claims three roles."""
        import re
        pat = re.compile(role_pattern("ffn_gate_exps.weight"))
        assert pat.search("blk.7.ffn_gate_exps.weight")
        assert not pat.search("blk.7.ffn_gate_shexp.weight")
        assert not pat.search("blk.7.ffn_gate_inp.weight")

    def test_anchored_matches_exactly_one_tensor(self):
        import re
        pat = re.compile(anchored("blk.3.ffn_down_exps.weight"))
        assert pat.search("blk.3.ffn_down_exps.weight")
        assert not pat.search("blk.30.ffn_down_exps.weight")

    def test_rendered_file_is_only_assignments(self):
        """llama-quantize does `while (file >> arg) parse_tensor_type(arg)`.

        Every whitespace-separated token must be pattern=TYPE — a comment line
        aborts the run with `malformed tensor type '#'`.
        """
        result = plan([Group("x", role_pattern("x.weight"), 1_000_000)], ladder(), 10**6)
        body = render_tensor_type_file(result)
        assert body.strip()
        for token in body.split():
            assert token.count("=") == 1, f"{token!r} is not pattern=TYPE"
            pattern, _, qtype = token.partition("=")
            assert pattern and qtype in QUANT_TYPES


class TestRegroup:
    def test_energy_is_parameter_weighted(self):
        gs = [
            Group("r", role_pattern("r"), 100, energy=1.0),
            Group("r", role_pattern("r"), 300, energy=5.0),
        ]
        merged = regroup_by_role(gs)[0]
        assert merged.n_params == 400
        assert merged.n_tensors == 2
        assert merged.energy == pytest.approx((100 * 1.0 + 300 * 5.0) / 400)


class TestTensorNames:
    def test_layer_and_role_split(self):
        t = Tensor("blk.7.ffn_down_exps.weight", (512, 2048, 133), "Q3_K", 0)
        assert t.layer == 7
        assert t.role == "ffn_down_exps.weight"
        assert t.n_params == 512 * 2048 * 133

    def test_model_level_tensor_has_no_layer(self):
        t = Tensor("token_embd.weight", (2048, 248320), "Q8_0", 0)
        assert t.layer is None
        assert t.role == "token_embd.weight"


def _write_gguf(path, tensors, kv=None):
    """Minimal GGUF writer, for exercising the reader."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    kv = kv or {}
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for key, val in kv.items():
        out += s(key) + struct.pack("<I", 8) + s(val)
    for name, shape, type_id in tensors:
        out += s(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<I", type_id) + struct.pack("<Q", 0)
    path.write_bytes(out)


class TestGGUFReader:
    def test_reads_metadata_and_directory(self, tmp_path):
        p = tmp_path / "m.gguf"
        _write_gguf(p, [("blk.0.attn_q.weight", (2048, 8192), 11)],
                    {"general.architecture": "qwen35"})
        g = read_header(p)
        assert g.architecture == "qwen35"
        assert g.tensors[0].type_name == "Q3_K"
        assert g.tensors[0].n_params == 2048 * 8192

    def test_rejects_a_non_gguf_file(self, tmp_path):
        p = tmp_path / "nope.bin"
        p.write_bytes(b"NOTGGUF!" + b"\x00" * 64)
        with pytest.raises(ValueError, match="not a GGUF"):
            read_header(p)

    def test_rejects_an_unknown_tensor_type(self, tmp_path):
        p = tmp_path / "m.gguf"
        _write_gguf(p, [("x.weight", (16,), 31)])  # 31 is a removed repacking variant
        with pytest.raises(ValueError, match="unknown ggml type"):
            read_header(p)


class TestTiers:
    def test_parse_size_distinguishes_binary_from_decimal(self):
        assert parse_size("1GiB") == 1024**3
        assert parse_size("1GB") == 1000**3
        assert parse_size("8.5G") == int(8.5 * 1024**3)
        assert parse_size("1024") == 1024

    def test_every_tier_leaves_headroom(self):
        for tier in TIERS.values():
            assert tier.weight_budget_bytes < tier.usable_bytes, tier.name
            assert tier.headroom_bytes > 0

    def test_bandwidth_bound_tier_declares_its_bandwidth(self):
        for tier in TIERS.values():
            if tier.bound == "bandwidth":
                assert tier.bandwidth_gibs is not None, tier.name


class TestBudgetResolution:
    """Tier presets expand to flags; explicit flags win over the preset."""

    @staticmethod
    def _args(**kw):
        from argparse import Namespace

        base = dict(max_bytes=None, backend=None, reserve=None, with_spec=False)
        base.update(kw)
        return Namespace(**base)

    def test_tier_supplies_budget_and_backend(self):
        from crucible.commands.plan import _resolve_budget
        from crucible.tiers import get

        tier = get("mac-mini-m4-16gb")
        budget, backend, exclude, reserved = _resolve_budget(self._args(), tier)
        assert budget == tier.weight_budget_bytes
        assert backend == "metal"
        assert reserved == 0

    def test_explicit_flags_override_the_preset(self):
        from crucible.commands.plan import _resolve_budget
        from crucible.tiers import get

        budget, backend, _, _ = _resolve_budget(
            self._args(max_bytes="4GiB", backend="cpu"), get("mac-mini-m4-16gb")
        )
        assert budget == 4 * GIB
        assert backend == "cpu"

    def test_tier_exclusions_reach_the_ladder(self):
        from crucible.commands.plan import _resolve_budget
        from crucible.tiers import get

        _, _, exclude, _ = _resolve_budget(self._args(), get("strix-halo-128gb"))
        assert "Q2_K" in exclude
        assert "Q2_K" not in {q.name for q in ladder(exclude=exclude)}

    def test_reserve_comes_off_the_weight_budget(self):
        from crucible.commands.plan import _resolve_budget

        budget, _, _, reserved = _resolve_budget(
            self._args(max_bytes="10GiB", reserve="500MB"), None
        )
        assert reserved == 500 * 1000**2
        assert budget == 10 * GIB - reserved

    def test_with_spec_reserves_the_tiers_draft_head(self):
        """A draft head competes with the weights it accelerates."""
        from crucible.commands.plan import _resolve_budget
        from crucible.tiers import get

        tier = get("mac-mini-m4-16gb")
        assert tier.spec_overhead_bytes > 0
        budget, _, _, reserved = _resolve_budget(self._args(with_spec=True), tier)
        assert reserved == tier.spec_overhead_bytes
        assert budget == tier.weight_budget_bytes - reserved

    def test_with_spec_is_free_on_a_bandwidth_bound_tier(self):
        from crucible.commands.plan import _resolve_budget
        from crucible.tiers import get

        _, _, _, reserved = _resolve_budget(self._args(with_spec=True), get("strix-halo-128gb"))
        assert reserved == 0

    def test_no_budget_at_all_is_an_error(self):
        from crucible.commands.plan import _resolve_budget

        with pytest.raises(SystemExit, match="need a budget"):
            _resolve_budget(self._args(), None)

    def test_reserving_the_whole_budget_is_an_error(self):
        from crucible.commands.plan import _resolve_budget

        with pytest.raises(SystemExit, match="nothing left for weights"):
            _resolve_budget(self._args(max_bytes="400MB", reserve="500MB"), None)


class TestSensitivityTable:
    """Cross-role allocation from measured KL-divergence rather than a prior."""

    @staticmethod
    def _write(tmp_path, **over):
        import json

        doc = {
            "baseline_type": "Q6_K",
            "probe_type": "Q3_K",
            "baseline_kld": 0.001,
            "roles": {
                "attn_qkv.weight": {"n_params": 500_000_000, "delta_kld": 0.050},
                "ssm_out.weight": {"n_params": 250_000_000, "delta_kld": 0.020},
                "ffn_down_exps.weight": {"n_params": 5_000_000_000, "delta_kld": 0.010},
            },
        }
        doc.update(over)
        p = tmp_path / "sens.json"
        p.write_text(json.dumps(doc))
        return p

    def test_coefficients_normalize_to_weighted_mean_one(self, tmp_path):
        """A table redistributes budget; it must not silently rescale the total."""
        from crucible.sensitivity import load

        t = load(self._write(tmp_path))
        total = sum(r.n_params for r in t.roles.values())
        mean = sum(r.coefficient * r.n_params for r in t.roles.values()) / total
        assert mean == pytest.approx(1.0)

    def test_high_delta_per_param_ranks_highest(self, tmp_path):
        """attn_qkv moved KLD 5x more than the experts with 1/10 the parameters."""
        from crucible.sensitivity import load

        t = load(self._write(tmp_path))
        assert t.coefficient("attn_qkv.weight") > t.coefficient("ssm_out.weight")
        assert t.coefficient("ssm_out.weight") > t.coefficient("ffn_down_exps.weight")

    def test_unmeasured_role_is_neutral(self, tmp_path):
        from crucible.sensitivity import load

        assert load(self._write(tmp_path)).coefficient("attn_gate.weight") == 1.0
        assert load(self._write(tmp_path)).coefficient(None) == 1.0

    def test_probe_must_be_worse_than_baseline(self, tmp_path):
        """Otherwise the denominator is <= 0 and every coefficient is meaningless."""
        from crucible.sensitivity import load

        with pytest.raises(ValueError, match="not more distorting"):
            load(self._write(tmp_path, baseline_type="Q3_K", probe_type="Q6_K"))

    def test_malformed_table_raises_rather_than_defaulting(self, tmp_path):
        """A table that silently means nothing is worse than no table at all."""
        import json

        from crucible.sensitivity import load

        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"baseline_type": "Q6_K", "probe_type": "Q3_K"}))
        with pytest.raises(ValueError, match="missing 'baseline_kld'"):
            load(p)

    def test_empty_roles_raises(self, tmp_path):
        from crucible.sensitivity import load

        with pytest.raises(ValueError, match="no roles"):
            load(self._write(tmp_path, roles={}))

    def test_sensitivity_moves_bits_between_roles(self, tmp_path):
        """The whole point: two equal-size groups diverge on measured sensitivity."""
        from crucible.sensitivity import load

        t = load(self._write(tmp_path))
        gs = [
            Group("attn_qkv.weight", role_pattern("attn_qkv.weight"), 100_000_000,
                  role="attn_qkv.weight", sensitivity=t.coefficient("attn_qkv.weight")),
            Group("ffn_down_exps.weight", role_pattern("ffn_down_exps.weight"), 100_000_000,
                  role="ffn_down_exps.weight",
                  sensitivity=t.coefficient("ffn_down_exps.weight")),
        ]
        result = plan(gs, ladder(), 120_000_000)
        chosen = {c.group.key: c.quant.bpw for c in result.choices}
        assert chosen["attn_qkv.weight"] > chosen["ffn_down_exps.weight"]
