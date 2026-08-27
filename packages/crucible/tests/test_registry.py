"""The method registry and the CLI wiring that reads it.

None of this loads a model — the point is that the plumbing between a flag
and an implementation is checkable without an 80 GB checkpoint on disk.
"""

import pytest

from crucible.cli import build_parser
from crucible.methods.registry import (
    METHOD_REGISTRY,
    SCORER_REGISTRY,
    get_method,
    get_scorer,
    method_names,
    scorer_names,
)
from crucible.types import MethodContext, ModelAttrs


def _attrs() -> ModelAttrs:
    return ModelAttrs(
        model_class="Fake",
        router="router",
        experts="experts",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        fused_gate_up=False,
        num_experts_key="num_experts",
        num_experts_per_tok_key="num_experts_per_tok",
    )


def _ctx(**overrides) -> MethodContext:
    base = dict(
        model=None, tokenizer=None, dataloader=None, attrs=_attrs(),
        num_experts=128, top_k=8, num_to_keep=80, ratio=0.375,
    )
    base.update(overrides)
    return MethodContext(**base)


class TestRegistryContents:
    def test_validated_path_is_registered(self):
        assert "reap" in method_names()
        assert "reap" in scorer_names()

    def test_every_method_is_callable(self):
        for method in METHOD_REGISTRY.values():
            assert callable(method.compress)
            assert method.summary

    def test_every_scorer_is_callable(self):
        for scorer in SCORER_REGISTRY.values():
            assert callable(scorer.score)
            assert scorer.summary

    def test_unknown_names_name_the_alternatives(self):
        # A typo'd --method should say what exists, not just fail.
        with pytest.raises(ValueError, match="reap"):
            get_method("raep")
        with pytest.raises(ValueError, match="reap"):
            get_scorer("raep")

    def test_ream_scores_itself(self):
        # REAM observes per layer as it merges, so the pipeline must not burn
        # a separate calibration pass whose output nobody reads.
        assert get_method("ream").uses_scores is False
        assert get_method("reap").uses_scores is True

    def test_reap_refuses_to_prune_without_scores(self):
        with pytest.raises(ValueError, match="scores"):
            get_method("reap").compress(_ctx(scores=None))


class TestReapThroughTheRegistry:
    """Score then prune a mock MoE the way `crucible compress` does it."""

    def _model_and_data(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from tests.test_observer import BATCH, SEQ_LEN, _Gemma4Config, _Gemma4Model

        torch.manual_seed(0)
        model = _Gemma4Model()
        model.config = _Gemma4Config()
        model.eval()

        ids = torch.randint(0, 100, (4 * BATCH, SEQ_LEN))

        def collate(batch):
            stacked = torch.stack([b[0] for b in batch])
            return {"input_ids": stacked, "attention_mask": torch.ones_like(stacked)}

        return model, DataLoader(TensorDataset(ids), batch_size=BATCH, collate_fn=collate)

    def test_scorer_then_method_prunes_the_model(self):
        from tests.test_observer import GEMMA4_ATTRS, NUM_EXPERTS, TOP_K

        model, dataloader = self._model_and_data()
        ctx = _ctx(
            model=model, dataloader=dataloader, attrs=GEMMA4_ATTRS,
            num_experts=NUM_EXPERTS, top_k=TOP_K, num_to_keep=NUM_EXPERTS - 1,
            ratio=1 / NUM_EXPERTS,
        )

        scoring = get_scorer("reap").score(ctx)
        ctx.scores = scoring.scores
        ctx.per_layer_keep = scoring.per_layer_keep
        info = get_method("reap").compress(ctx).info

        assert scoring.observation is not None
        assert model.config.num_experts == NUM_EXPERTS - 1
        assert all(len(kept) == NUM_EXPERTS - 1 for kept in info["experts_kept"])
        assert info["per_layer_keep"] is None  # uniform


class TestCliWiring:
    def test_method_choices_come_from_the_registry(self):
        parser = build_parser()
        args = parser.parse_args(["compress", "some/model", "--method", "ream"])

        assert args.method == "ream"
        assert args.run is not None

    def test_methods_contribute_their_own_flags(self):
        # --group-size is REAM's, declared by the registry entry rather than
        # hard-coded in the compress command.
        parser = build_parser()
        args = parser.parse_args(
            ["compress", "some/model", "--method", "ream", "--group-size", "8"]
        )

        assert args.group_size == 8

    def test_scorers_contribute_their_own_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            ["compress", "some/model", "--scoring", "task-aware", "--task-alpha", "2.0"]
        )

        assert args.scoring == "task-aware"
        assert args.task_alpha == 2.0

    def test_unregistered_method_is_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compress", "some/model", "--method", "nope"])

    def test_every_subcommand_dispatches(self):
        parser = build_parser()
        for argv in (
            ["inspect", "some/model"],
            ["observe", "some/model"],
            ["quantize", "some/model"],
        ):
            assert callable(parser.parse_args(argv).run)

    def test_crucible_does_not_benchmark(self):
        # Measuring a served model is crucible-bench's job, in its own package
        # with its own dependency tree. If a benchmark subcommand reappears
        # here it has brought an endpoint client into a package whose reason
        # for existing is producing weights.
        parser = build_parser()
        for argv in (
            ["eval", "some/model"],
            ["bench", "--model", "label"],
            ["compare", "a.json", "b.json"],
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(argv)




class TestUnsupportedModelGuidance:
    """The registry miss is the most likely first failure a new user sees."""

    def test_names_the_architecture_it_rejected(self):
        from crucible.models.registry import get_model_attrs

        with pytest.raises(ValueError, match="LlamaForCausalLM"):
            get_model_attrs("LlamaForCausalLM")

    def test_lists_what_is_supported(self):
        from crucible.models.registry import MODEL_REGISTRY, get_model_attrs

        with pytest.raises(ValueError) as e:
            get_model_attrs("NotAModel")
        for name in MODEL_REGISTRY:
            assert name in str(e.value)

    def test_says_how_to_add_one(self):
        # An error that only reports the miss sends the user to the source to
        # work out whether adding support is a one-liner or a fork.
        from crucible.models.registry import get_model_attrs

        with pytest.raises(ValueError) as e:
            get_model_attrs("NotAModel")
        msg = str(e.value)
        assert "ModelAttrs" in msg and "registry.py" in msg

    def test_warns_about_the_architecture_unwrapping_trap(self):
        # AutoModelForCausalLM on a *ForConditionalGeneration checkpoint reports
        # the inner class, so the name in the error can look unfamiliar and send
        # someone hunting for a bug that isn't there.
        from crucible.models.registry import get_model_attrs

        with pytest.raises(ValueError, match="ForConditionalGeneration"):
            get_model_attrs("NotAModel")
