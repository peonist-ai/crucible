"""Tests for crucible.methods.task_aware."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from crucible.methods.task_aware import (
    _boost_layer,
    compute_task_aware_scores,
    compute_task_specificity,
)
from crucible.types import ExpertScore
from tests.test_observer import (
    BATCH,
    GEMMA4_ATTRS,
    NUM_EXPERTS,
    SEQ_LEN,
    _Gemma4Config,
    _Gemma4Model,
)


def _make_dataloader(num_batches=3, seed=42):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 100, (num_batches * BATCH, SEQ_LEN), generator=gen)
    masks = torch.ones_like(ids)
    ds = TensorDataset(ids, masks)

    def collate(batch):
        return {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
        }

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


class TestBoostLayer:
    def test_alpha_zero_returns_task_scores(self):
        """With alpha=0, boosted scores equal task scores."""
        task = [ExpertScore(0, i, score=float(i + 1)) for i in range(4)]
        general = [ExpertScore(0, i, score=float(4 - i)) for i in range(4)]

        boosted = _boost_layer(task, general, alpha=0.0)

        for b, t in zip(boosted, task):
            assert b.score == t.score

    def test_specialists_boosted(self):
        """Task specialists (high task score, low general) should be boosted."""
        # Expert 0: pure task specialist (task=10, general=0)
        # Expert 1: pure generalist (task=5, general=5)
        # Expert 2: general-only (task=0, general=10)
        # Frequencies must clear _boost_layer's min_freq guard (1% of tokens),
        # which deliberately neutralizes the ratio for rarely-activated experts.
        task = [
            ExpertScore(0, 0, score=10.0, frequency=0.5),
            ExpertScore(0, 1, score=5.0, frequency=0.5),
            ExpertScore(0, 2, score=0.0, frequency=0.5),
        ]
        general = [
            ExpertScore(0, 0, score=0.0, frequency=0.5),
            ExpertScore(0, 1, score=5.0, frequency=0.5),
            ExpertScore(0, 2, score=10.0, frequency=0.5),
        ]

        boosted = _boost_layer(task, general, alpha=2.0)

        # Specialist should be boosted (R=1.0, boost = 1 + 2*(1-0.5) = 2.0)
        assert boosted[0].score == 10.0 * 2.0
        # Generalist stays neutral (R=0.5, boost = 1 + 2*(0.5-0.5) = 1.0)
        assert boosted[1].score == 5.0 * 1.0
        # General-only is penalized but score was 0 anyway
        assert boosted[2].score == 0.0

    def test_boost_preserves_metadata(self):
        task = [ExpertScore(5, 3, score=2.0, frequency=0.1, activation_norm=1.5, router_weight=0.3)]
        general = [ExpertScore(5, 3, score=1.0)]

        boosted = _boost_layer(task, general, alpha=1.0)

        assert boosted[0].layer_idx == 5
        assert boosted[0].expert_idx == 3
        assert boosted[0].frequency == 0.1
        assert boosted[0].activation_norm == 1.5
        assert boosted[0].router_weight == 0.3


class TestTaskSpecificity:
    def test_pure_specialist(self):
        task = [[ExpertScore(0, 0, score=10.0)]]
        general = [[ExpertScore(0, 0, score=0.0)]]
        ratios = compute_task_specificity(task, general)
        assert ratios[0][0] == 1.0

    def test_pure_generalist(self):
        task = [[ExpertScore(0, 0, score=5.0)]]
        general = [[ExpertScore(0, 0, score=5.0)]]
        ratios = compute_task_specificity(task, general)
        assert ratios[0][0] == 0.5

    def test_zero_scores(self):
        task = [[ExpertScore(0, 0, score=0.0)]]
        general = [[ExpertScore(0, 0, score=0.0)]]
        ratios = compute_task_specificity(task, general)
        assert ratios[0][0] == 0.5  # default for zero/zero


class TestEndToEnd:
    def test_dual_observation(self):
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()

        task_dl = _make_dataloader(seed=42)
        general_dl = _make_dataloader(seed=99)

        scores = compute_task_aware_scores(
            model, task_dl, general_dl, GEMMA4_ATTRS, alpha=1.0
        )

        assert len(scores) == 2  # NUM_LAYERS
        for layer_scores in scores:
            assert len(layer_scores) == NUM_EXPERTS
            for s in layer_scores:
                assert s.score >= 0

    def test_higher_alpha_changes_ranking(self):
        """Higher alpha should change expert ranking vs alpha=0."""
        torch.manual_seed(42)
        model = _Gemma4Model()
        model.config = _Gemma4Config()

        task_dl = _make_dataloader(seed=42)
        general_dl = _make_dataloader(seed=99)

        scores_0 = compute_task_aware_scores(
            model, task_dl, general_dl, GEMMA4_ATTRS, alpha=0.0
        )
        scores_2 = compute_task_aware_scores(
            model, task_dl, general_dl, GEMMA4_ATTRS, alpha=2.0
        )

        # Scores should differ with different alpha
        vals_0 = [s.score for s in scores_0[0]]
        vals_2 = [s.score for s in scores_2[0]]
        assert vals_0 != vals_2
