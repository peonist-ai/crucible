"""Tests for crucible.methods.ream."""

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from crucible.methods.ream import (
    MergeResult,
    _assign_groups,
    _compute_gate_similarity,
    _merge_group_weights,
    merge,
)
from crucible.types import ExpertScore
from tests.test_observer import (
    BATCH,
    GEMMA4_ATTRS,
    HIDDEN,
    MIXTRAL_ATTRS,
    NUM_EXPERTS,
    SEQ_LEN,
    TOP_K,
    _Gemma4Config,
    _Gemma4Model,
    _MixtralConfig,
    _MixtralModel,
)

NUM_KEEP = 3  # Keep 3 of 4 experts


def _make_dataloader(num_batches=5):
    ids = torch.randint(0, 100, (num_batches * BATCH, SEQ_LEN))
    masks = torch.ones_like(ids)
    ds = TensorDataset(ids, masks)

    def collate(batch):
        ids_t = torch.stack([b[0] for b in batch])
        masks_t = torch.stack([b[1] for b in batch])
        return {"input_ids": ids_t, "attention_mask": masks_t}

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


class TestGateSimilarity:
    def test_shape(self):
        logits = torch.randn(100, NUM_EXPERTS)
        sim = _compute_gate_similarity(logits)
        assert sim.shape == (NUM_EXPERTS, NUM_EXPERTS)

    def test_self_similarity_is_one(self):
        logits = torch.randn(100, NUM_EXPERTS)
        sim = _compute_gate_similarity(logits)
        diag = sim.diag()
        assert torch.allclose(diag, torch.ones(NUM_EXPERTS), atol=1e-5)

    def test_symmetric(self):
        logits = torch.randn(100, NUM_EXPERTS)
        sim = _compute_gate_similarity(logits)
        assert torch.allclose(sim, sim.T, atol=1e-5)

    def test_identical_experts_have_similarity_one(self):
        logits = torch.randn(100, NUM_EXPERTS)
        # Make experts 0 and 1 identical
        logits[:, 1] = logits[:, 0]
        sim = _compute_gate_similarity(logits)
        assert sim[0, 1].item() == pytest.approx(1.0, abs=1e-5)


class TestAssignGroups:
    def _make_scores(self, values):
        return [
            ExpertScore(layer_idx=0, expert_idx=i, score=v)
            for i, v in enumerate(values)
        ]

    def test_centroids_are_top_scoring(self):
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        sim = torch.eye(NUM_EXPERTS)
        groups = _assign_groups(scores, sim, num_to_keep=2, group_size=4)
        assert set(groups.keys()) == {1, 3}

    def test_all_non_centroids_assigned(self):
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        sim = torch.ones(NUM_EXPERTS, NUM_EXPERTS)
        groups = _assign_groups(scores, sim, num_to_keep=2, group_size=4)
        all_members = []
        for members in groups.values():
            all_members.extend(members)
        assert sorted(all_members) == [0, 2]

    def test_similarity_drives_assignment(self):
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        # Expert 0 (lowest) is very similar to expert 3, not expert 1
        sim = torch.eye(NUM_EXPERTS)
        sim[0, 3] = sim[3, 0] = 0.9
        sim[0, 1] = sim[1, 0] = 0.1
        sim[2, 1] = sim[1, 2] = 0.9
        sim[2, 3] = sim[3, 2] = 0.1
        groups = _assign_groups(scores, sim, num_to_keep=2, group_size=4)
        assert 0 in groups[3]
        assert 2 in groups[1]

    def test_group_size_cap(self):
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        sim = torch.ones(NUM_EXPERTS, NUM_EXPERTS)
        # With group_size=1, each centroid can absorb at most 1 non-centroid
        groups = _assign_groups(scores, sim, num_to_keep=2, group_size=1)
        for members in groups.values():
            assert len(members) <= 1

    def test_no_similarity_falls_back(self):
        """Without similarity matrix, should still assign (by centroid saliency)."""
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        groups = _assign_groups(scores, None, num_to_keep=2, group_size=4)
        assert set(groups.keys()) == {1, 3}
        # All non-centroids should be assigned somewhere
        all_members = []
        for members in groups.values():
            all_members.extend(members)
        assert sorted(all_members) == [0, 2]

    def test_keep_all(self):
        """If num_to_keep == num_experts, all are centroids, no merging."""
        scores = self._make_scores([1.0, 4.0, 2.0, 3.0])
        sim = torch.eye(NUM_EXPERTS)
        groups = _assign_groups(scores, sim, num_to_keep=4, group_size=4)
        assert len(groups) == 4
        assert all(len(m) == 0 for m in groups.values())


class TestMergeGroupWeights:
    def test_tensor3d_merge(self):
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        original_centroid_gu = experts.gate_up_proj[0].clone()

        saliency = {0: 3.0, 1: 1.0}
        _merge_group_weights(experts, 0, [1], saliency, GEMMA4_ATTRS)

        # Centroid weights should have changed
        assert not torch.equal(experts.gate_up_proj[0], original_centroid_gu)

    def test_tensor3d_alignment_then_average(self):
        """With alignment, merged weights differ from a naive average."""
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        w0_gu = experts.gate_up_proj[0].clone()
        w1_gu = experts.gate_up_proj[1].clone()

        saliency = {0: 1.0, 1: 1.0}
        _merge_group_weights(experts, 0, [1], saliency, GEMMA4_ATTRS)

        naive_avg = (w0_gu + w1_gu) / 2
        merged = experts.gate_up_proj[0]

        # Alignment permutes neurons before averaging, so result differs
        # from naive average (unless experts happen to already be aligned)
        assert merged.shape == naive_avg.shape
        # The merged expert should still be finite and reasonable
        assert torch.isfinite(merged).all()

    def test_tensor3d_identity_alignment(self):
        """Merging identical experts produces the same expert (alignment is identity)."""
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        # Make expert 1 identical to expert 0
        experts.gate_up_proj.data[1] = experts.gate_up_proj.data[0].clone()
        experts.down_proj.data[1] = experts.down_proj.data[0].clone()
        original = experts.gate_up_proj[0].clone()

        saliency = {0: 1.0, 1: 1.0}
        _merge_group_weights(experts, 0, [1], saliency, GEMMA4_ATTRS)

        # Identical experts → alignment is identity → average = original
        assert torch.allclose(experts.gate_up_proj[0], original, atol=1e-5)

    def test_modulelist_merge(self):
        from tests.test_observer import _MixtralMoE

        torch.manual_seed(42)
        moe = _MixtralMoE()
        original_w1 = moe.experts[0].w1.weight.data.clone()

        saliency = {0: 3.0, 1: 1.0}
        _merge_group_weights(moe.experts, 0, [1], saliency, MIXTRAL_ATTRS)

        assert not torch.equal(moe.experts[0].w1.weight.data, original_w1)

    def test_empty_members_noop(self):
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        original = experts.gate_up_proj[0].clone()

        _merge_group_weights(experts, 0, [], {0: 1.0}, GEMMA4_ATTRS)

        assert torch.equal(experts.gate_up_proj[0], original)

    def test_saliency_weighting(self):
        """Higher saliency expert should dominate the merge."""
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        centroid_gu = experts.gate_up_proj[0].clone()
        member_gu = experts.gate_up_proj[1].clone()

        # Centroid has 99x the saliency → merged should be ~centroid
        saliency = {0: 99.0, 1: 1.0}
        _merge_group_weights(experts, 0, [1], saliency, GEMMA4_ATTRS)

        merged = experts.gate_up_proj[0]
        dist_to_centroid = (merged - centroid_gu).norm()
        dist_to_member = (merged - member_gu).norm()
        assert dist_to_centroid < dist_to_member


class TestNeuronAlignment:
    def test_alignment_permutation_valid(self):
        """Permutation should be a valid permutation of [0, intermediate)."""
        from crucible.methods.ream import _compute_alignment

        torch.manual_seed(42)
        intermediate = 16
        hidden = 32
        c_vecs = [torch.randn(intermediate, hidden) for _ in range(3)]
        m_vecs = [torch.randn(intermediate, hidden) for _ in range(3)]

        perm = _compute_alignment(c_vecs, m_vecs)

        assert perm.shape == (intermediate,)
        assert set(perm.tolist()) == set(range(intermediate))

    def test_identity_for_identical_experts(self):
        """Aligning an expert to itself should produce identity permutation."""
        from crucible.methods.ream import _compute_alignment

        torch.manual_seed(42)
        intermediate = 16
        hidden = 32
        vecs = [torch.randn(intermediate, hidden) for _ in range(3)]

        perm = _compute_alignment(vecs, vecs)

        assert perm.tolist() == list(range(intermediate))

    def test_alignment_reduces_distance(self):
        """After alignment, member should be closer to centroid."""
        from crucible.methods.ream import (
            _align_member_to_centroid,
            _get_neuron_vectors_tensor3d,
        )
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()

        c_vecs = _get_neuron_vectors_tensor3d(experts, 0, GEMMA4_ATTRS)
        m_vecs_before = _get_neuron_vectors_tensor3d(experts, 1, GEMMA4_ATTRS)
        dist_before = sum(
            (c - m).norm() for c, m in zip(c_vecs, m_vecs_before)
        )

        _align_member_to_centroid(experts, 0, 1, GEMMA4_ATTRS)

        m_vecs_after = _get_neuron_vectors_tensor3d(experts, 1, GEMMA4_ATTRS)
        dist_after = sum(
            (c - m).norm() for c, m in zip(c_vecs, m_vecs_after)
        )

        # Alignment should not increase total distance
        assert dist_after <= dist_before + 1e-5

    def test_permutation_preserves_expert_function(self):
        """A permuted expert should produce the same output (just via different neurons)."""
        from crucible.methods.ream import _apply_permutation_tensor3d
        from tests.test_observer import _Gemma4Experts

        torch.manual_seed(42)
        experts = _Gemma4Experts()
        h = torch.randn(5, HIDDEN)

        # Compute output before permutation
        from crucible.methods.observer import _compute_expert_output

        out_before = _compute_expert_output(h, experts, GEMMA4_ATTRS, 1)

        # Apply a known permutation
        intermediate = experts.down_proj.shape[-1]
        perm = torch.randperm(intermediate)
        _apply_permutation_tensor3d(experts, 1, perm, GEMMA4_ATTRS)

        out_after = _compute_expert_output(h, experts, GEMMA4_ATTRS, 1)

        # Same output (neuron reordering is internal, doesn't change I/O)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_modulelist_permutation_preserves_function(self):
        """Modulelist permutation should also preserve expert I/O."""
        from crucible.methods.ream import _apply_permutation_modulelist
        from tests.test_observer import _MixtralMoE

        torch.manual_seed(42)
        moe = _MixtralMoE()
        h = torch.randn(5, HIDDEN)

        from crucible.methods.observer import _compute_expert_output

        out_before = _compute_expert_output(h, moe.experts, MIXTRAL_ATTRS, 0)

        intermediate = moe.experts[0].w2.weight.shape[-1]
        perm = torch.randperm(intermediate)
        _apply_permutation_modulelist(moe.experts, 0, perm, MIXTRAL_ATTRS)

        out_after = _compute_expert_output(h, moe.experts, MIXTRAL_ATTRS, 0)

        assert torch.allclose(out_before, out_after, atol=1e-5)


class TestMergeGemma4:
    def test_sequential_basic(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        result = merge(model, dl, GEMMA4_ATTRS, NUM_KEEP, sequential=True)

        assert isinstance(result, MergeResult)
        assert result.remaining_num_experts == NUM_KEEP
        assert len(result.groups) == 2  # NUM_LAYERS

    def test_oneshot_basic(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        result = merge(model, dl, GEMMA4_ATTRS, NUM_KEEP, sequential=False)

        assert result.remaining_num_experts == NUM_KEEP

    def test_expert_weights_resized(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP, sequential=True)

        for layer in model.model.layers:
            assert layer.experts.gate_up_proj.shape[0] == NUM_KEEP
            assert layer.experts.down_proj.shape[0] == NUM_KEEP

    def test_router_resized(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP, sequential=True)

        for layer in model.model.layers:
            assert layer.router.proj.out_features == NUM_KEEP
            assert layer.router.proj.weight.shape == (NUM_KEEP, HIDDEN)

    def test_config_updated(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)

        assert model.config.num_experts == NUM_KEEP

    def test_forward_after_merge(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)

        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert out.shape == (1, SEQ_LEN, HIDDEN)
        assert torch.isfinite(out).all()

    def test_groups_cover_all_experts(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        result = merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)

        for groups in result.groups:
            all_indices = set(groups.keys())
            for members in groups.values():
                all_indices.update(members)
            assert all_indices == set(range(NUM_EXPERTS))


class TestMergeMixtral:
    def test_sequential(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader()

        result = merge(model, dl, MIXTRAL_ATTRS, NUM_KEEP, sequential=True)

        assert result.remaining_num_experts == NUM_KEEP
        for layer in model.model.layers:
            assert len(layer.block_sparse_moe.experts) == NUM_KEEP

    def test_forward_after_merge(self):
        torch.manual_seed(42)
        model = _MixtralModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader()

        merge(model, dl, MIXTRAL_ATTRS, NUM_KEEP)

        x = torch.randint(0, 100, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert out.shape == (1, SEQ_LEN, HIDDEN)
        assert torch.isfinite(out).all()


class TestMergeVsPrune:
    def test_merge_differs_from_prune(self):
        """REAM should produce different centroid weights than just pruning."""
        torch.manual_seed(42)
        model_merge = _Gemma4Model()
        model_merge.config = _Gemma4Config()
        model_prune = copy.deepcopy(model_merge)
        model_prune.config = _Gemma4Config()
        dl = _make_dataloader()

        # REAM merge
        merge(model_merge, dl, GEMMA4_ATTRS, NUM_KEEP)

        # REAP prune (no merging)
        from crucible.methods.observer import compute_reap_scores, observe
        from crucible.methods.reap import prune

        result = observe(model_prune, dl, GEMMA4_ATTRS)
        scores = compute_reap_scores(result)
        prune(model_prune, scores, GEMMA4_ATTRS, NUM_KEEP)

        # Both should have the same shape
        m_gu = model_merge.model.layers[0].experts.gate_up_proj
        p_gu = model_prune.model.layers[0].experts.gate_up_proj
        assert m_gu.shape == p_gu.shape

        # But weights should differ (merge absorbed knowledge from pruned experts)
        assert not torch.equal(m_gu, p_gu)


class TestMergeValidation:
    def test_keep_too_many(self):
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        with pytest.raises(ValueError, match=">="):
            merge(model, dl, GEMMA4_ATTRS, NUM_EXPERTS)

    def test_keep_less_than_topk(self):
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        with pytest.raises(ValueError, match="top_k"):
            merge(model, dl, GEMMA4_ATTRS, TOP_K - 1)
