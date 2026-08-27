"""Tests for crucible.methods.reap."""

from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from crucible.methods.observer import compute_reap_scores, observe
from crucible.methods.reap import (
    GroupedRouting,
    PruneResult,
    _resolve_grouped_routing,
    _select_experts,
    prune,
)
from crucible.types import ExpertScore, ModelAttrs

# Reuse mock models from observer tests
from tests.test_observer import (
    BATCH,
    GEMMA4_ATTRS,
    HIDDEN,
    INTERMEDIATE,
    MIXTRAL_ATTRS,
    NUM_EXPERTS,
    QWEN35_ATTRS,
    SEQ_LEN,
    TOP_K,
    _Gemma4Config,
    _Gemma4Model,
    _MixtralConfig,
    _MixtralModel,
    _Qwen35Config,
    _Qwen35Model,
)

NUM_KEEP = 3  # Keep 3 of 4 experts (25% pruning)


def _make_dataloader(num_batches=5):
    ids = torch.randint(0, 100, (num_batches * BATCH, SEQ_LEN))
    masks = torch.ones_like(ids)
    ds = TensorDataset(ids, masks)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": masks}

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


def _observe_and_score(model, attrs):
    torch.manual_seed(42)
    dl = _make_dataloader()
    result = observe(model, dl, attrs)
    return compute_reap_scores(result)


class TestSelectExperts:
    def test_keeps_highest_scores(self):
        scores = [
            ExpertScore(layer_idx=0, expert_idx=0, score=1.0),
            ExpertScore(layer_idx=0, expert_idx=1, score=4.0),
            ExpertScore(layer_idx=0, expert_idx=2, score=2.0),
            ExpertScore(layer_idx=0, expert_idx=3, score=3.0),
        ]
        keep, remove = _select_experts(scores, num_to_keep=2)
        assert keep == [1, 3]  # highest scores
        assert remove == [0, 2]  # lowest scores

    def test_indices_sorted(self):
        scores = [
            ExpertScore(layer_idx=0, expert_idx=i, score=float(i))
            for i in range(NUM_EXPERTS)
        ]
        keep, remove = _select_experts(scores, num_to_keep=NUM_KEEP)
        assert keep == sorted(keep)
        assert remove == sorted(remove)

    def test_keep_all_minus_one(self):
        scores = [
            ExpertScore(layer_idx=0, expert_idx=i, score=float(i))
            for i in range(NUM_EXPERTS)
        ]
        keep, remove = _select_experts(scores, num_to_keep=NUM_EXPERTS - 1)
        assert len(keep) == NUM_EXPERTS - 1
        assert len(remove) == 1
        assert remove[0] == 0  # lowest score


class TestPruneGemma4:
    def test_basic(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        result = prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        assert isinstance(result, PruneResult)
        assert result.original_num_experts == NUM_EXPERTS
        assert result.remaining_num_experts == NUM_KEEP
        assert len(result.experts_kept) == 2  # NUM_LAYERS
        assert all(len(k) == NUM_KEEP for k in result.experts_kept)

    def test_expert_weights_resized(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        for layer in model.model.layers:
            gate_up = layer.experts.gate_up_proj
            down = layer.experts.down_proj
            assert gate_up.shape[0] == NUM_KEEP
            assert gate_up.shape[1] == 2 * INTERMEDIATE
            assert gate_up.shape[2] == HIDDEN
            assert down.shape[0] == NUM_KEEP
            assert down.shape[1] == HIDDEN
            assert down.shape[2] == INTERMEDIATE

    def test_router_resized(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        for layer in model.model.layers:
            proj = layer.router.proj
            assert proj.out_features == NUM_KEEP
            assert proj.weight.shape == (NUM_KEEP, HIDDEN)

    def test_config_updated(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        assert model.config.num_experts == NUM_KEEP

    def test_forward_after_prune(self):
        """Model should still produce valid output after pruning."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        # Forward pass should work with pruned model
        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert out.shape == (1, SEQ_LEN, HIDDEN)
        assert torch.isfinite(out).all()

    def test_correct_experts_removed(self):
        """The lowest-scoring experts should be the ones removed."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        result = prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        for layer_idx, (layer_scores, removed) in enumerate(
            zip(scores, result.experts_removed)
        ):
            # The removed expert should have the lowest score
            removed_scores = [
                s.score for s in layer_scores if s.expert_idx in removed
            ]
            kept_scores = [
                s.score
                for s in layer_scores
                if s.expert_idx not in removed
            ]
            assert max(removed_scores) <= min(kept_scores)


class TestPruneMixtral:
    def test_basic(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        scores = _observe_and_score(model, MIXTRAL_ATTRS)

        result = prune(model, scores, MIXTRAL_ATTRS, num_experts_to_keep=NUM_KEEP)

        assert result.remaining_num_experts == NUM_KEEP

    def test_experts_modulelist_resized(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        scores = _observe_and_score(model, MIXTRAL_ATTRS)

        prune(model, scores, MIXTRAL_ATTRS, num_experts_to_keep=NUM_KEEP)

        for layer in model.model.layers:
            assert len(layer.block_sparse_moe.experts) == NUM_KEEP

    def test_router_resized(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        scores = _observe_and_score(model, MIXTRAL_ATTRS)

        prune(model, scores, MIXTRAL_ATTRS, num_experts_to_keep=NUM_KEEP)

        for layer in model.model.layers:
            gate = layer.block_sparse_moe.gate
            assert gate.out_features == NUM_KEEP
            assert gate.weight.shape == (NUM_KEEP, HIDDEN)

    def test_forward_after_prune(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        scores = _observe_and_score(model, MIXTRAL_ATTRS)

        prune(model, scores, MIXTRAL_ATTRS, num_experts_to_keep=NUM_KEEP)

        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert out.shape == (1, SEQ_LEN, HIDDEN)
        assert torch.isfinite(out).all()


class TestExpertParamNames:
    """Which stacked tensors get sliced on a tensor3d model.

    Fused models name one tensor for gate and up; unfused name two. Slicing a
    hardcoded (gate, down) pair silently skipped `up_proj` on an unfused 3D
    model — the tensors then disagree on expert count.
    """

    def test_fused_yields_two_names(self):
        from crucible.methods.reap import _expert_param_names

        assert _expert_param_names(GEMMA4_ATTRS) == ["gate_up_proj", "down_proj"]

    def test_unfused_yields_all_three(self):
        from dataclasses import replace

        from crucible.methods.reap import _expert_param_names

        attrs = replace(
            GEMMA4_ATTRS,
            fused_gate_up=False,
            gate_proj="gate_proj",
            up_proj="up_proj",
        )
        assert _expert_param_names(attrs) == ["gate_proj", "up_proj", "down_proj"]

    def test_unfused_tensor3d_slices_up_proj(self):
        """The regression itself: every stacked tensor ends up the same width."""
        from dataclasses import replace

        import torch.nn as nn

        from crucible.methods.reap import _expert_param_names, _prune_experts

        attrs = replace(
            GEMMA4_ATTRS,
            fused_gate_up=False,
            gate_proj="gate_proj",
            up_proj="up_proj",
        )

        class _Unfused(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN)
                )
                self.up_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN)
                )
                self.down_proj = nn.Parameter(
                    torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
                )

        layer = nn.Module()
        layer.experts = _Unfused()
        _prune_experts(layer, [0, 2], attrs)

        for name in _expert_param_names(attrs):
            assert getattr(layer.experts, name).shape[0] == 2, name


class TestPruneValidation:
    def test_keep_too_many(self):
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        with pytest.raises(ValueError, match="must be <"):
            prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_EXPERTS)

    def test_keep_zero(self):
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        # Keeping zero experts is caught by the top_k floor — the router always
        # selects top_k, so any k below it is invalid and 0 is the extreme case.
        with pytest.raises(ValueError, match="top_k"):
            prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=0)

    def test_keep_less_than_topk(self):
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        scores = _observe_and_score(model, GEMMA4_ATTRS)

        with pytest.raises(ValueError, match="top_k"):
            prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=TOP_K - 1)


class TestNonUniformPadding:
    """Non-uniform budgets pad layers back to a rectangle.

    Regression coverage: padding used to require a Linear router, so it raised on
    Qwen3.5's Parameter-based router — every non-uniform budget was unreachable
    on the model family crucible actually targets.
    """

    def _scored(self, model_cls, config_cls, attrs):
        torch.manual_seed(42)
        model = model_cls()
        model.config = config_cls()
        return model, _observe_and_score(model, attrs)

    def test_parameter_router_pads(self):
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)

        with pytest.warns(RuntimeWarning, match="cloning experts"):
            result = prune(model, scores, QWEN35_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        assert result.remaining_num_experts == NUM_KEEP
        assert model.config.num_experts == NUM_KEEP
        for layer in model.model.layers:
            assert layer.mlp.gate.weight.shape == (NUM_KEEP, HIDDEN)
            assert layer.mlp.experts.gate_up_proj.shape[0] == NUM_KEEP
            assert layer.mlp.experts.down_proj.shape[0] == NUM_KEEP

    def test_parameter_router_forward_after_pad(self):
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)

        with pytest.warns(RuntimeWarning):
            prune(model, scores, QWEN35_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert torch.isfinite(out).all()

    def test_bias_less_router_clones_rather_than_zeroes(self):
        """A zero-weight pad row is selectable — its logit is 0, and real logits
        go negative. So padded experts must be real experts, not zero holes."""
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)

        with pytest.warns(RuntimeWarning):
            result = prune(model, scores, QWEN35_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        assert result.pad_strategy == "cloned"
        gate = model.model.layers[1].mlp.gate.weight
        experts = model.model.layers[1].mlp.experts

        # Nothing about the pad is zeroed...
        assert gate[-1].abs().sum() > 0
        assert experts.gate_up_proj[-1].abs().sum() > 0
        # ...because it is an exact twin of a surviving expert, in both the
        # router row and the FFN weights.
        twins = [
            i for i in range(NUM_KEEP - 1) if torch.equal(gate[-1], gate[i])
        ]
        assert twins, "padded router row is not a copy of any kept expert"
        assert torch.equal(
            experts.gate_up_proj[-1], experts.gate_up_proj[twins[0]]
        )

    def test_biased_router_masks_padded_experts(self):
        """With an additive bias available, padding stays inert: zero weights and
        a logit pinned far below every real expert."""
        model, scores = self._scored(_Gemma4Model, _Gemma4Config, GEMMA4_ATTRS)
        for layer in model.model.layers:
            layer.router.proj.bias = torch.nn.Parameter(
                torch.zeros(NUM_EXPERTS, dtype=layer.router.proj.weight.dtype)
            )

        result = prune(model, scores, GEMMA4_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        assert result.pad_strategy == "masked"
        padded = model.model.layers[1].router.proj
        assert padded.bias[-1].item() < -1e3
        assert padded.weight[-1].abs().sum().item() == 0

    def test_per_expert_scale_tracks_the_router(self):
        """Gemma 4's per-expert scale is [num_experts] — it has to be sliced on
        prune and padded on pad, or it desynchronises from the router rows."""
        model, scores = self._scored(_Gemma4Model, _Gemma4Config, GEMMA4_ATTRS)

        with pytest.warns(RuntimeWarning):
            prune(model, scores, GEMMA4_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        for layer in model.model.layers:
            router = layer.router
            assert router.per_expert_scale.shape == (NUM_KEEP,)
            assert router.proj.weight.shape[0] == router.per_expert_scale.shape[0]

    def test_uniform_prune_slices_per_expert_scale(self):
        model, scores = self._scored(_Gemma4Model, _Gemma4Config, GEMMA4_ATTRS)
        prune(model, scores, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)
        for layer in model.model.layers:
            assert layer.router.per_expert_scale.shape == (NUM_KEEP,)

    def test_modulelist_padding_supported(self):
        """Padding used to raise NotImplementedError for ModuleList experts."""
        model, scores = self._scored(_MixtralModel, _MixtralConfig, MIXTRAL_ATTRS)

        with pytest.warns(RuntimeWarning):
            prune(model, scores, MIXTRAL_ATTRS, [NUM_KEEP, NUM_KEEP - 1])

        for layer in model.model.layers:
            assert len(layer.block_sparse_moe.experts) == NUM_KEEP
        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert torch.isfinite(out).all()

    def test_score_correction_bias_enables_masking(self):
        """A DeepSeek-style score-correction bias is additive and per-expert, so
        it can mask padded rows even though the router has no Linear bias."""
        from dataclasses import replace

        attrs = replace(QWEN35_ATTRS, router_score_bias="e_score_correction_bias")
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)
        for layer in model.model.layers:
            layer.mlp.gate.e_score_correction_bias = torch.nn.Parameter(
                torch.zeros(NUM_EXPERTS)
            )

        result = prune(model, scores, attrs, [NUM_KEEP, NUM_KEEP - 1])

        assert result.pad_strategy == "masked"
        for layer in model.model.layers:
            bias = layer.mlp.gate.e_score_correction_bias
            assert bias.shape == (NUM_KEEP,)
        padded = model.model.layers[1].mlp.gate
        assert padded.e_score_correction_bias[-1].item() < -1e3
        assert padded.weight[-1].abs().sum().item() == 0

    def test_score_correction_bias_sliced_on_uniform_prune(self):
        from dataclasses import replace

        attrs = replace(QWEN35_ATTRS, router_score_bias="e_score_correction_bias")
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)
        for layer in model.model.layers:
            layer.mlp.gate.e_score_correction_bias = torch.nn.Parameter(
                torch.arange(NUM_EXPERTS, dtype=torch.float32)
            )

        result = prune(model, scores, attrs, num_experts_to_keep=NUM_KEEP)

        for li, layer in enumerate(model.model.layers):
            bias = layer.mlp.gate.e_score_correction_bias
            assert bias.shape == (NUM_KEEP,)
            # Values must follow the surviving experts, not just be truncated.
            assert bias.tolist() == [float(i) for i in result.experts_kept[li]]

    def test_uniform_budget_does_not_pad(self):
        model, scores = self._scored(_Qwen35Model, _Qwen35Config, QWEN35_ATTRS)
        result = prune(model, scores, QWEN35_ATTRS, num_experts_to_keep=NUM_KEEP)
        assert result.pad_strategy is None


# ---------------------------------------------------------------------------
# Group-limited routing
# ---------------------------------------------------------------------------

GROUPED_ATTRS = ModelAttrs(
    model_class="GroupedMoeForCausalLM",
    router="mlp.gate",
    experts="mlp.experts",
    gate_proj="gate_up_proj",
    up_proj="gate_up_proj",
    down_proj="down_proj",
    fused_gate_up=True,
    num_experts_key="num_experts",
    num_experts_per_tok_key="num_experts_per_tok",
    expert_storage="tensor3d",
    n_group_key="n_group",
    top_k_group_key="topk_group",
)

GROUPED_EXPERTS = 8


@dataclass
class _GroupedConfig:
    num_experts: int = GROUPED_EXPERTS
    num_experts_per_tok: int = 2
    n_group: int = 2
    topk_group: int = 1


def _grouped_scores(per_expert: list[float]) -> list[list[ExpertScore]]:
    return [
        [
            ExpertScore(layer_idx=0, expert_idx=i, score=s)
            for i, s in enumerate(per_expert)
        ]
    ]


class TestGroupedRouting:
    def test_resolves_from_config(self):
        g = _resolve_grouped_routing(_GroupedConfig(), GROUPED_ATTRS, GROUPED_EXPERTS)
        assert g == GroupedRouting(n_group=2, top_k_group=1, group_size=4)

    def test_absent_when_registry_declares_none(self):
        assert (
            _resolve_grouped_routing(_Gemma4Config(), GEMMA4_ATTRS, NUM_EXPERTS)
            is None
        )

    def test_keeps_equal_count_per_group(self):
        # Group 0 (experts 0-3) holds every high score; a global ranking would
        # empty group 1 and break the router's group selection.
        scores = _grouped_scores([9.0, 8.0, 7.0, 6.0, 1.0, 0.9, 0.8, 0.7])
        grouping = GroupedRouting(n_group=2, top_k_group=1, group_size=4)

        keep, remove = _select_experts(scores[0], 4, grouping=grouping)

        assert keep == [0, 1, 4, 5]
        assert remove == [2, 3, 6, 7]

    def test_ranks_by_saliency_within_group(self):
        scores = _grouped_scores([1.0, 5.0, 2.0, 4.0, 1.0, 9.0, 2.0, 8.0])
        grouping = GroupedRouting(n_group=2, top_k_group=1, group_size=4)

        keep, _ = _select_experts(scores[0], 4, grouping=grouping)

        assert keep == [1, 3, 5, 7]

    def test_rejects_budget_not_divisible_by_groups(self):
        from crucible.methods.reap import _validate_grouped_keep

        grouping = GroupedRouting(n_group=2, top_k_group=1, group_size=4)
        with pytest.raises(ValueError, match="divisible by the router's group"):
            _validate_grouped_keep(5, 2, grouping, "Layer 0")

    def test_rejects_budget_that_starves_selected_groups(self):
        from crucible.methods.reap import _validate_grouped_keep

        # top_k_group=1 group of 1 expert each cannot supply top_k=2.
        grouping = GroupedRouting(n_group=4, top_k_group=1, group_size=4)
        with pytest.raises(ValueError, match="fewer than top_k"):
            _validate_grouped_keep(4, 2, grouping, "Layer 0")

    def test_accepts_budget_that_satisfies_top_k(self):
        from crucible.methods.reap import _validate_grouped_keep

        grouping = GroupedRouting(n_group=2, top_k_group=1, group_size=4)
        _validate_grouped_keep(4, 2, grouping, "Layer 0")  # 2 per group >= top_k


class TestEndToEnd:
    def test_observe_prune_reobserve(self):
        """Full pipeline: observe, prune, then observe again on pruned model."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        # First observation
        result1 = observe(model, dl, GEMMA4_ATTRS)
        scores1 = compute_reap_scores(result1)
        assert result1.num_experts == NUM_EXPERTS

        # Prune
        prune(model, scores1, GEMMA4_ATTRS, num_experts_to_keep=NUM_KEEP)

        # Second observation on pruned model
        result2 = observe(model, dl, GEMMA4_ATTRS)
        scores2 = compute_reap_scores(result2)

        assert result2.num_experts == NUM_KEEP
        for layer_scores in scores2:
            assert len(layer_scores) == NUM_KEEP
            # All experts should have indices 0..NUM_KEEP-1 now
            indices = {s.expert_idx for s in layer_scores}
            assert indices == set(range(NUM_KEEP))
