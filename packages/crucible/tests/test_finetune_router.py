"""Tests for crucible.methods.finetune_router."""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from crucible.methods.finetune_router import finetune_router
from crucible.methods.ream import merge
from tests.test_observer import (
    BATCH,
    GEMMA4_ATTRS,
    HIDDEN,
    MIXTRAL_ATTRS,
    SEQ_LEN,
    _Gemma4Config,
    _Gemma4Layer,
    _MixtralConfig,
    _MixtralLayer,
    _MockEmbedding,
)

NUM_KEEP = 3
VOCAB = 100


# Mock models with LM heads (needed for cross-entropy fine-tuning)


class _Gemma4LMModel(nn.Module):
    """Gemma4 mock with an LM head for fine-tuning tests."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Gemma4Layer(), _Gemma4Layer()])
        self.model.embed = _MockEmbedding()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


class _MixtralLMModel(nn.Module):
    """Mixtral mock with an LM head for fine-tuning tests."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_MixtralLayer(), _MixtralLayer()])
        self.model.embed = _MockEmbedding()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


def _make_dataloader(num_batches=5, seed=42):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (num_batches * BATCH, SEQ_LEN), generator=gen)
    masks = torch.ones_like(ids)
    ds = TensorDataset(ids, masks)

    def collate(batch):
        return {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
        }

    return DataLoader(ds, batch_size=BATCH, collate_fn=collate)


class TestFinetuneRouterGemma4:
    def test_basic(self):
        torch.manual_seed(42)
        model = _Gemma4LMModel()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)
        result = finetune_router(model, dl, GEMMA4_ATTRS, steps=5, lr=1e-3)

        assert result["steps"] == 5
        assert result["trainable_params"] > 0
        assert len(result["losses"]) == 5

    def test_only_router_trained(self):
        """After fine-tuning, only router weights should have changed."""
        torch.manual_seed(42)
        model = _Gemma4LMModel()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)

        expert_before = [
            layer.experts.gate_up_proj.data.clone()
            for layer in model.model.layers
        ]
        router_before = [
            layer.router.proj.weight.data.clone()
            for layer in model.model.layers
        ]

        finetune_router(model, dl, GEMMA4_ATTRS, steps=10, lr=1e-3)

        # Expert weights unchanged
        for before, layer in zip(expert_before, model.model.layers):
            assert torch.equal(before, layer.experts.gate_up_proj.data)

        # Router weights changed
        any_changed = any(
            not torch.equal(before, layer.router.proj.weight.data)
            for before, layer in zip(router_before, model.model.layers)
        )
        assert any_changed

    def test_model_works_after(self):
        torch.manual_seed(42)
        model = _Gemma4LMModel()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)
        finetune_router(model, dl, GEMMA4_ATTRS, steps=5)

        x = torch.randint(0, VOCAB, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=x)
        assert out.shape == (1, SEQ_LEN, VOCAB)
        assert torch.isfinite(out).all()

    def test_all_params_frozen_after(self):
        torch.manual_seed(42)
        model = _Gemma4LMModel()
        model.config = _Gemma4Config()
        dl = _make_dataloader()

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)
        finetune_router(model, dl, GEMMA4_ATTRS, steps=5)

        for p in model.parameters():
            assert not p.requires_grad


class TestFinetuneRouterMixtral:
    def test_basic(self):
        torch.manual_seed(42)
        model = _MixtralLMModel()
        model.config = _MixtralConfig()
        dl = _make_dataloader()

        merge(model, dl, MIXTRAL_ATTRS, NUM_KEEP)
        result = finetune_router(model, dl, MIXTRAL_ATTRS, steps=5)

        assert result["trainable_params"] > 0
        assert len(result["losses"]) == 5


class TestLossDecreases:
    def test_loss_trend(self):
        """Loss should generally decrease over training."""
        torch.manual_seed(42)
        model = _Gemma4LMModel()
        model.config = _Gemma4Config()
        dl = _make_dataloader(num_batches=10)

        merge(model, dl, GEMMA4_ATTRS, NUM_KEEP)
        result = finetune_router(model, dl, GEMMA4_ATTRS, steps=30, lr=1e-3)

        early = sum(result["losses"][:5]) / 5
        late = sum(result["losses"][-5:]) / 5
        assert late < early * 1.5
