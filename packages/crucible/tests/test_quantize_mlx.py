"""Tests for crucible.quantize_mlx.

Covers the parts that do not need mlx-lm installed: the bit-allocation policy and
its composition with the model's own predicate. `require_mlx`, `save` and the
GPTQ hand-off are thin wrappers over the optional dependency and are exercised by
running the command.

The composition tests are the point of the file. `mlx_lm.utils.quantize_model`
resolves its predicate with a plain `or`, so supplying one of our own silently
replaces whatever the model asked for — and on Qwen 3.5/3.6 MoE what the model
asks for is 8-bit routers.
"""

import math

import pytest

from crucible.quantize_mlx import (
    compose_with_model_predicate,
    crucible_quant_predicate,
    estimate_size_gb,
    resolve_module_bits,
)


class _Array:
    def __init__(self, shape):
        self.shape = shape
        self.size = math.prod(shape)


class _Quantizable:
    """Stands in for an nn.Linear / SwitchLinear: has weight and to_quantized."""

    def __init__(self, shape):
        self.weight = _Array(shape)

    def to_quantized(self, **kwargs):  # pragma: no cover - presence is the point
        raise NotImplementedError


class _NotQuantizable:
    def __init__(self, shape):
        self.weight = _Array(shape)


class _Model:
    """A model exposing its own quant_predicate, as mlx-lm's Qwen3.5 does."""

    def __init__(self, predicate):
        self.quant_predicate = predicate


def _crucible():
    return crucible_quant_predicate(
        bits=4, group_size=64, high_bits=8, high_group_size=64
    )


# Mirrors mlx_lm/models/qwen3_5.py: routers and the shared-expert gate held at 8
# bits, everything else left to the caller's defaults.
def _qwen35_predicate(path, module):
    if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
        return {"group_size": 64, "bits": 8}
    return True


ATTENTION = "language_model.model.layers.7.self_attn.q_proj"
EXPERTS = "language_model.model.layers.7.mlp.switch_mlp.gate_proj"
ROUTER = "language_model.model.layers.7.mlp.gate"
SHARED_GATE = "language_model.model.layers.7.mlp.shared_expert_gate"
SHARED_EXPERT = "language_model.model.layers.7.mlp.shared_expert.down_proj"


class TestCruciblePredicate:
    def test_attention_is_held_high(self):
        assert _crucible()(ATTENTION, _Quantizable((2048, 2048))) == {
            "group_size": 64,
            "bits": 8,
        }

    @pytest.mark.parametrize(
        "path",
        ["language_model.model.embed_tokens", "language_model.lm_head"],
    )
    def test_embeddings_and_output_are_held_high(self, path):
        assert _crucible()(path, _Quantizable((248320, 2048)))["bits"] == 8

    @pytest.mark.parametrize("path", [EXPERTS, SHARED_EXPERT])
    def test_expert_weight_takes_the_default(self, path):
        # True means "use the command's --bits", which is where the size goes.
        assert _crucible()(path, _Quantizable((133, 512, 2048))) is True

    def test_opting_out_leaves_everything_default(self):
        predicate = crucible_quant_predicate(
            bits=4,
            group_size=64,
            high_bits=8,
            high_group_size=64,
            keep_attention_high=False,
            keep_embeddings_high=False,
        )
        assert predicate(ATTENTION, _Quantizable((2048, 2048))) is True

    def test_it_does_not_decide_router_width(self):
        # Routers are the model's business; ours must fall through so that
        # composition can hand them back. If this ever returns a dict, the
        # composition test below stops proving anything.
        assert _crucible()(ROUTER, _Quantizable((133, 2048))) is True


class TestComposition:
    def test_model_protection_survives_our_predicate(self):
        """The regression this module exists for.

        Passing a predicate to quantize_model replaces the model's outright. If
        composition is dropped, the router falls from 8 bits to --bits, which
        changes which expert runs rather than merely how well it runs.
        """
        composed = compose_with_model_predicate(
            _Model(_qwen35_predicate), _crucible()
        )
        assert composed(ROUTER, _Quantizable((133, 2048)))["bits"] == 8
        assert composed(SHARED_GATE, _Quantizable((1, 2048)))["bits"] == 8

    def test_our_split_still_applies_where_the_model_is_neutral(self):
        composed = compose_with_model_predicate(
            _Model(_qwen35_predicate), _crucible()
        )
        assert composed(ATTENTION, _Quantizable((2048, 2048)))["bits"] == 8
        assert composed(EXPERTS, _Quantizable((133, 512, 2048))) is True

    def test_model_veto_is_honoured(self):
        composed = compose_with_model_predicate(
            _Model(lambda path, module: False), _crucible()
        )
        assert composed(ATTENTION, _Quantizable((2048, 2048))) is False

    def test_no_model_predicate_passes_ours_through(self):
        composed = compose_with_model_predicate(object(), _crucible())
        assert composed(ATTENTION, _Quantizable((2048, 2048)))["bits"] == 8

    def test_no_predicate_of_ours_leaves_the_model_in_charge(self):
        composed = compose_with_model_predicate(_Model(_qwen35_predicate), None)
        assert composed(ROUTER, _Quantizable((133, 2048)))["bits"] == 8

    def test_both_absent_is_none(self):
        assert compose_with_model_predicate(object(), None) is None


class TestResolveModuleBits:
    def test_non_quantizable_modules_are_skipped(self):
        assert (
            resolve_module_bits(
                "norm", _NotQuantizable((2048,)), None, bits=4, group_size=64
            )
            is None
        )

    def test_indivisible_last_dim_is_skipped_silently(self):
        # mlx-lm's own guard. It logs nothing, so a module can land at full
        # precision without any sign in the output — which is why plan_quantization
        # reports the count.
        assert (
            resolve_module_bits(
                EXPERTS, _Quantizable((133, 512, 100)), None, bits=4, group_size=64
            )
            is None
        )

    def test_dict_decision_overrides_both_bits_and_group(self):
        predicate = lambda path, module: {"bits": 6, "group_size": 32}  # noqa: E731
        assert resolve_module_bits(
            ATTENTION, _Quantizable((2048, 2048)), predicate, bits=4, group_size=64
        ) == (6, 32)

    def test_true_decision_takes_the_defaults(self):
        assert resolve_module_bits(
            EXPERTS, _Quantizable((133, 512, 2048)), None, bits=3, group_size=64
        ) == (3, 64)

    def test_false_decision_leaves_it_alone(self):
        predicate = lambda path, module: False  # noqa: E731
        assert (
            resolve_module_bits(
                ATTENTION, _Quantizable((2048, 2048)), predicate, bits=4, group_size=64
            )
            is None
        )


def test_size_estimate_matches_hand_arithmetic():
    # 19.17B params at 4.5 bits/weight — 4-bit affine plus a bf16 scale and a
    # bf16 bias per group of 64 — is what the 16GB Mac has to hold.
    assert estimate_size_gb(19_173_552_768, 4.5) == pytest.approx(10.04, abs=0.02)
