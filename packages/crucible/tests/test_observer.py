"""Tests for crucible.methods.observer."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from crucible.methods.observer import (
    ObservationResult,
    _compute_expert_output,
    _find_layers,
    _get_config_value,
    _parse_router_output,
    _resolve_expert_activation,
    compute_reap_scores,
    observe,
)
from crucible.types import ModelAttrs

# ---------------------------------------------------------------------------
# Minimal mock MoE models
# ---------------------------------------------------------------------------

HIDDEN = 32
INTERMEDIATE = 16
NUM_EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 2
SEQ_LEN = 8
BATCH = 2


class _MockEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, HIDDEN)

    def forward(self, input_ids):
        return self.embed(input_ids)


# --- Gemma4-style: tensor3d experts, tuple-returning router ---


class _Gemma4Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        # Gemma 4's router carries a per-expert scale. How the real one uses it
        # is not modelled here — what matters for pruning is that it is a
        # [num_experts] tensor that has to be sliced and padded in step with the
        # router rows, and the mock covers that bookkeeping.
        self.per_expert_scale = nn.Parameter(torch.ones(NUM_EXPERTS))

    def forward(self, hidden_states):
        logits = self.proj(hidden_states) * self.per_expert_scale
        probs = F.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, TOP_K, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return probs, top_k_weights, top_k_index


class _Gemma4Experts(nn.Module):
    def __init__(self):
        super().__init__()
        # Real HF fused-expert modules carry their activation as `act_fn`; the
        # observer reads it instead of assuming SiLU.
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.02
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for k in range(TOP_K):
            idx = top_k_index[..., k]
            w = top_k_weights[..., k].unsqueeze(-1)
            for eidx in idx.unique():
                mask = idx == eidx
                h = hidden_states[mask]
                gu = F.linear(h, self.gate_up_proj[eidx])
                g, u = gu.chunk(2, dim=-1)
                expert_out = F.linear(F.silu(g) * u, self.down_proj[eidx])
                out[mask] += w[mask] * expert_out
        return out


class _Gemma4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Gemma4Router()
        self.experts = _Gemma4Experts()
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, hidden_states):
        h = self.norm(hidden_states)
        h = h.view(-1, HIDDEN)
        probs, weights, indices = self.router(h)
        out = self.experts(h, indices, weights)
        return hidden_states + out.view_as(hidden_states)


class _Gemma4Model(nn.Module):
    """Minimal mock of Gemma4ForConditionalGeneration."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Gemma4Layer() for _ in range(NUM_LAYERS)]
        )
        self.model.embed = _MockEmbedding()

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


@dataclass
class _Gemma4Config:
    num_experts: int = NUM_EXPERTS
    top_k_experts: int = TOP_K


GEMMA4_ATTRS = ModelAttrs(
    model_class="Gemma4ForConditionalGeneration",
    router="router",
    experts="experts",
    gate_proj="gate_up_proj",
    up_proj="gate_up_proj",
    down_proj="down_proj",
    fused_gate_up=True,
    num_experts_key="num_experts",
    num_experts_per_tok_key="top_k_experts",
    expert_storage="tensor3d",
)


# --- Mixtral-style: modulelist experts, logits-returning router ---


class _MixtralExpertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.w3 = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.w2 = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, hidden_states):
        # A real expert module owns its forward; the observer calls this rather
        # than reconstructing the FFN from the projection names.
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class _MixtralMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList(
            [_MixtralExpertFFN() for _ in range(NUM_EXPERTS)]
        )

    def forward(self, hidden_states):
        h = hidden_states.view(-1, HIDDEN)
        logits = self.gate(h)
        weights = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(weights, TOP_K, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(h)
        for k in range(TOP_K):
            for eidx in top_i[:, k].unique():
                mask = top_i[:, k] == eidx
                expert_out = self.experts[eidx](h[mask])
                out[mask] += top_w[:, k][mask].unsqueeze(-1) * expert_out
        return out.view_as(hidden_states)


class _MixtralLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = _MixtralMoE()
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, hidden_states):
        return hidden_states + self.block_sparse_moe(self.norm(hidden_states))


class _MixtralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_MixtralLayer() for _ in range(NUM_LAYERS)]
        )
        self.model.embed = _MockEmbedding()

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


@dataclass
class _MixtralConfig:
    num_local_experts: int = NUM_EXPERTS
    num_experts_per_tok: int = TOP_K
    # Real Mixtral configs carry this; it is the observer's fallback source for
    # the expert activation when the module exposes no act_fn.
    hidden_act: str = "silu"


MIXTRAL_ATTRS = ModelAttrs(
    model_class="MixtralForCausalLM",
    router="block_sparse_moe.gate",
    experts="block_sparse_moe.experts",
    gate_proj="w1",
    up_proj="w3",
    down_proj="w2",
    fused_gate_up=False,
    num_experts_key="num_local_experts",
    num_experts_per_tok_key="num_experts_per_tok",
    expert_storage="modulelist",
)


# --- Qwen3.5-style: tensor3d experts, Parameter-based router with no Linear
# child and no bias. This is the shape that used to break non-uniform pruning:
# `_prune_router` handled it, padding did not.


class _Qwen35Router(nn.Module):
    """Router that scores experts from a raw Parameter, like Qwen3_5MoeTopKRouter."""

    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, HIDDEN) * 0.02)
        self.num_experts = num_experts

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, TOP_K, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        return probs, top_w, top_i


class _Qwen35Experts(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.act_fn = F.silu
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, HIDDEN, INTERMEDIATE) * 0.02
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for k in range(top_k_index.shape[-1]):
            idx = top_k_index[..., k]
            w = top_k_weights[..., k].unsqueeze(-1)
            for eidx in idx.unique():
                mask = idx == eidx
                gu = F.linear(hidden_states[mask], self.gate_up_proj[eidx])
                g, u = gu.chunk(2, dim=-1)
                out[mask] += w[mask] * F.linear(
                    self.act_fn(g) * u, self.down_proj[eidx]
                )
        return out


class _Qwen35MoE(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.gate = _Qwen35Router(num_experts)
        self.experts = _Qwen35Experts(num_experts)

    def forward(self, hidden_states):
        h = hidden_states.view(-1, HIDDEN)
        _probs, weights, indices = self.gate(h)
        return self.experts(h, indices, weights).view_as(hidden_states)


class _Qwen35Layer(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.mlp = _Qwen35MoE(num_experts)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, hidden_states):
        return hidden_states + self.mlp(self.norm(hidden_states))


class _Qwen35Model(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Qwen35Layer(num_experts) for _ in range(NUM_LAYERS)]
        )
        self.model.embed = _MockEmbedding()

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


@dataclass
class _Qwen35Config:
    num_experts: int = NUM_EXPERTS
    num_experts_per_tok: int = TOP_K


QWEN35_ATTRS = ModelAttrs(
    model_class="Qwen3_5MoeForCausalLM",
    router="mlp.gate",
    experts="mlp.experts",
    gate_proj="gate_up_proj",
    up_proj="gate_up_proj",
    down_proj="down_proj",
    fused_gate_up=True,
    num_experts_key="num_experts",
    num_experts_per_tok_key="num_experts_per_tok",
    expert_storage="tensor3d",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataloader(num_batches=3):
    ids = torch.randint(0, 100, (num_batches * BATCH, SEQ_LEN))
    masks = torch.ones_like(ids)
    ds = TensorDataset(ids, masks)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": masks}

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_resolve_path(self):
        from crucible.methods.observer import _resolve_path

        model = _Gemma4Model()
        router = _resolve_path(model.model.layers[0], "router")
        assert isinstance(router, _Gemma4Router)

    def test_resolve_path_nested(self):
        from crucible.methods.observer import _resolve_path

        model = _MixtralModel()
        gate = _resolve_path(model.model.layers[0], "block_sparse_moe.gate")
        assert isinstance(gate, nn.Linear)

    def test_find_layers(self):
        model = _Gemma4Model()
        layers = _find_layers(model)
        assert len(layers) == NUM_LAYERS

    def test_get_config_value_direct(self):
        cfg = _Gemma4Config()
        assert _get_config_value(cfg, "num_experts") == NUM_EXPERTS

    def test_get_config_value_nested(self):
        @dataclass
        class _Outer:
            text_config: _Gemma4Config = None

        cfg = _Outer(text_config=_Gemma4Config())
        assert _get_config_value(cfg, "num_experts") == NUM_EXPERTS

    def test_parse_router_output_tuple(self):
        probs = torch.randn(4, NUM_EXPERTS)
        weights = torch.randn(4, TOP_K)
        indices = torch.randint(0, NUM_EXPERTS, (4, TOP_K))
        w, i = _parse_router_output((probs, weights, indices), TOP_K)
        assert torch.equal(w, weights)
        assert torch.equal(i, indices)

    def test_parse_router_output_logits(self):
        logits = torch.randn(4, NUM_EXPERTS)
        w, i = _parse_router_output(logits, TOP_K)
        assert w.shape == (4, TOP_K)
        assert i.shape == (4, TOP_K)
        # Weights should be valid probabilities
        assert (w >= 0).all()
        assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestExpertOutput:
    def test_tensor3d(self):
        experts = _Gemma4Experts()
        h = torch.randn(3, HIDDEN)
        out = _compute_expert_output(h, experts, GEMMA4_ATTRS, 0)
        assert out.shape == (3, HIDDEN)

    def test_modulelist(self):
        moe = _MixtralMoE()
        h = torch.randn(3, HIDDEN)
        out = _compute_expert_output(h, moe.experts, MIXTRAL_ATTRS, 0)
        assert out.shape == (3, HIDDEN)


class _BareExperts(nn.Module):
    """tensor3d experts that expose no `act_fn` — the ambiguous case."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.02
        )


@dataclass
class _GeluConfig:
    num_experts: int = NUM_EXPERTS
    top_k_experts: int = TOP_K
    hidden_act: str = "gelu"


class TestExpertActivation:
    """The activation is read off the model, never assumed.

    Only `tensor3d` storage needs this — there is no per-expert module to call,
    so the FFN is recomputed by hand and the activation is the one thing the
    weight shapes do not reveal. Guessing it is how REAM went wrong on Gemma 4.
    """

    def test_explicit_override_wins(self):
        from dataclasses import replace

        attrs = replace(GEMMA4_ATTRS, expert_act="gelu")
        # _Gemma4Experts carries act_fn=silu; the override must beat it.
        act = _resolve_expert_activation(_Gemma4Experts(), attrs, _Gemma4Config())
        assert act is F.gelu

    def test_module_act_fn_used_when_no_override(self):
        act = _resolve_expert_activation(
            _Gemma4Experts(), GEMMA4_ATTRS, _GeluConfig()
        )
        # act_fn on the module outranks the config.
        assert act is F.silu

    def test_falls_back_to_config_hidden_act(self):
        act = _resolve_expert_activation(
            _BareExperts(), GEMMA4_ATTRS, _GeluConfig()
        )
        assert act is F.gelu

    def test_reads_hidden_act_from_nested_text_config(self):
        @dataclass
        class _Multimodal:
            text_config: _GeluConfig = None

        act = _resolve_expert_activation(
            _BareExperts(), GEMMA4_ATTRS, _Multimodal(text_config=_GeluConfig())
        )
        assert act is F.gelu

    def test_warns_and_defaults_to_silu_when_unknowable(self):
        with pytest.warns(RuntimeWarning, match="assuming SiLU"):
            act = _resolve_expert_activation(_BareExperts(), GEMMA4_ATTRS, None)
        assert act is F.silu

    def test_unknown_activation_name_is_rejected(self):
        from dataclasses import replace

        attrs = replace(GEMMA4_ATTRS, expert_act="not_an_activation")
        with pytest.raises(ValueError, match="Unknown expert activation"):
            _resolve_expert_activation(_BareExperts(), attrs, None)

    def test_activation_actually_changes_the_output(self):
        """A GELU model must not be scored as if it were SiLU."""
        torch.manual_seed(0)
        experts = _BareExperts()
        h = torch.randn(5, HIDDEN)

        as_silu = _compute_expert_output(h, experts, GEMMA4_ATTRS, 0, F.silu)
        as_gelu = _compute_expert_output(h, experts, GEMMA4_ATTRS, 0, F.gelu)
        assert not torch.allclose(as_silu, as_gelu)

        # And the resolved-from-config path agrees with passing GELU explicitly.
        resolved = _resolve_expert_activation(experts, GEMMA4_ATTRS, _GeluConfig())
        assert torch.equal(
            _compute_expert_output(h, experts, GEMMA4_ATTRS, 0, resolved), as_gelu
        )


class TestModulelistUsesRealForward:
    def test_own_forward_is_called(self):
        """A real expert module's forward is used verbatim, so any activation or
        extra projection it applies is honoured without being re-declared."""
        moe = _MixtralMoE()
        h = torch.randn(4, HIDDEN)
        expected = moe.experts[0](h)
        assert torch.equal(
            _compute_expert_output(h, moe.experts, MIXTRAL_ATTRS, 0), expected
        )

    def test_falls_back_to_projections_without_a_forward(self):
        """Weight-holder classes with no forward still work, via the named
        projections and a resolved activation."""

        class _NoForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
                self.w3 = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
                self.w2 = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

        experts = nn.ModuleList([_NoForward()])
        h = torch.randn(3, HIDDEN)
        out = _compute_expert_output(h, experts, MIXTRAL_ATTRS, 0, F.silu)
        e = experts[0]
        assert torch.equal(out, e.w2(F.silu(e.w1(h)) * e.w3(h)))


class TestObserveGemma4:
    def test_basic(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=2)

        result = observe(model, dl, GEMMA4_ATTRS)

        assert isinstance(result, ObservationResult)
        assert len(result.layer_stats) == NUM_LAYERS
        assert result.num_experts == NUM_EXPERTS
        assert result.top_k == TOP_K
        assert result.total_tokens == 2 * BATCH * SEQ_LEN

    def test_all_experts_observed(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=10)

        result = observe(model, dl, GEMMA4_ATTRS)

        # With enough data, all experts should be selected at least once
        for layer_stats in result.layer_stats:
            assert (layer_stats.count > 0).all(), "Some experts were never selected"

    def test_counts_sum_to_expected(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=2)

        result = observe(model, dl, GEMMA4_ATTRS)

        total_tokens = result.total_tokens
        for layer_stats in result.layer_stats:
            # Each token selects TOP_K experts, so total count = tokens * top_k
            assert layer_stats.count.sum().item() == total_tokens * TOP_K

    def test_scores_positive(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=5)

        result = observe(model, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)

        assert len(scores) == NUM_LAYERS
        for layer_scores in scores:
            assert len(layer_scores) == NUM_EXPERTS
            for s in layer_scores:
                # Gate values are positive (softmax), norms are positive
                assert s.score >= 0
                assert s.frequency >= 0
                assert s.activation_norm >= 0
                assert s.router_weight >= 0

    def test_store_router_logits(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=2)

        result = observe(
            model, dl, GEMMA4_ATTRS, store_router_logits=True
        )

        for layer_stats in result.layer_stats:
            assert len(layer_stats.router_logits) > 0
            for logits in layer_stats.router_logits:
                assert logits.shape[-1] == NUM_EXPERTS


class TestObserveMixtral:
    def test_basic(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader(num_batches=2)

        result = observe(model, dl, MIXTRAL_ATTRS)

        assert len(result.layer_stats) == NUM_LAYERS
        assert result.num_experts == NUM_EXPERTS
        assert result.top_k == TOP_K

    def test_counts_sum_to_expected(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader(num_batches=2)

        result = observe(model, dl, MIXTRAL_ATTRS)

        for layer_stats in result.layer_stats:
            assert layer_stats.count.sum().item() == result.total_tokens * TOP_K

    def test_scores(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader(num_batches=5)

        result = observe(model, dl, MIXTRAL_ATTRS)
        scores = compute_reap_scores(result)

        assert len(scores) == NUM_LAYERS
        for layer_scores in scores:
            for s in layer_scores:
                assert s.score >= 0


class TestReapScores:
    def test_frequency_sums_to_topk(self):
        """Total frequency across experts should equal top_k (each token picks top_k)."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=5)

        result = observe(model, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)

        for layer_scores in scores:
            total_freq = sum(s.frequency for s in layer_scores)
            assert abs(total_freq - TOP_K) < 1e-6

    def test_score_equals_gate_times_norm_average(self):
        """Verify S_j = mean(g * ||f||) by checking against component averages."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=5)

        result = observe(model, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)

        for layer_idx, layer_stats in enumerate(result.layer_stats):
            for s in scores[layer_idx]:
                count = layer_stats.count[s.expert_idx].item()
                if count > 0:
                    expected = (
                        layer_stats.weighted_sum[s.expert_idx].item() / count
                    )
                    assert abs(s.score - expected) < 1e-10

    def test_zero_count_expert(self):
        """Experts with no selections should have zero scores."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        # Very few samples — some experts may not be selected
        dl = _make_dataloader(num_batches=1)

        result = observe(model, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)

        for layer_idx, layer_stats in enumerate(result.layer_stats):
            for s in scores[layer_idx]:
                if layer_stats.count[s.expert_idx].item() == 0:
                    assert s.score == 0.0
                    assert s.frequency == 0.0
                    assert s.activation_norm == 0.0
                    assert s.router_weight == 0.0

    def test_layer_indices_correct(self):
        """ExpertScore.layer_idx should match the actual model layer index."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=2)

        result = observe(model, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)

        for stat_idx, layer_scores in enumerate(scores):
            expected_layer_idx = result.moe_layer_indices[stat_idx]
            for s in layer_scores:
                assert s.layer_idx == expected_layer_idx
