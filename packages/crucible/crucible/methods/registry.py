"""Compression method registry — maps a strategy name to its implementation.

The sibling of `models/registry.py`: that one teaches crucible about a new
architecture, this one teaches it about a new way to compress. Adding a
method means adding an entry here — the CLI reads its `--method` and
`--scoring` choices, its per-method flags, and its dispatch out of these
registries, so it never learns any individual method's name.

Compression runs in two halves:

  scorer  → ranks experts (and may plan a non-uniform per-layer budget)
  method  → mutates the model using that ranking

They are registered separately because they compose: any scorer can feed any
method that consumes scores. A method that scores internally (REAM observes
per layer as it merges) sets `uses_scores=False` and the pipeline skips
scoring entirely.

The adapters below keep their heavy imports inside the function body. This
module is imported to build the argument parser, long before any model is
loaded, and importing torch there would cost seconds on `crucible --help`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from crucible.types import MethodContext, MethodResult, ScoringResult

# Both registries hand out an argparse parser (or argument group) to let an
# entry declare its own flags.
AddArguments = Callable[[argparse.ArgumentParser], None]


class ScoreFn(Protocol):
    """Rank experts for one model. Must not mutate the model."""

    def __call__(self, ctx: MethodContext) -> ScoringResult: ...


class CompressFn(Protocol):
    """Compress the model in `ctx` in-place and describe what it did."""

    def __call__(self, ctx: MethodContext) -> MethodResult: ...


@dataclass(frozen=True)
class Scorer:
    """Registry entry for an expert-scoring strategy."""

    name: str
    summary: str
    score: ScoreFn
    add_arguments: AddArguments | None = None


@dataclass(frozen=True)
class Method:
    """Registry entry for a compression method.

    uses_scores: whether the pipeline should run a scorer first and hand the
        result over in `ctx.scores`. False for methods that observe on their
        own — scoring them anyway would burn a full calibration pass whose
        output nobody reads.
    """

    name: str
    summary: str
    compress: CompressFn
    uses_scores: bool = True
    add_arguments: AddArguments | None = None


SCORER_REGISTRY: dict[str, Scorer] = {}
METHOD_REGISTRY: dict[str, Method] = {}


def register_scorer(scorer: Scorer) -> Scorer:
    """Add a scoring strategy. Later registrations replace earlier ones."""
    SCORER_REGISTRY[scorer.name] = scorer
    return scorer


def register_method(method: Method) -> Method:
    """Add a compression method. Later registrations replace earlier ones."""
    METHOD_REGISTRY[method.name] = method
    return method


def get_scorer(name: str) -> Scorer:
    if name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scoring strategy: {name}. "
            f"Available: {', '.join(SCORER_REGISTRY)}"
        )
    return SCORER_REGISTRY[name]


def get_method(name: str) -> Method:
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown compression method: {name}. "
            f"Available: {', '.join(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[name]


def scorer_names() -> list[str]:
    return list(SCORER_REGISTRY)


def method_names() -> list[str]:
    return list(METHOD_REGISTRY)


# ---------------------------------------------------------------------------
# Scoring strategies
# ---------------------------------------------------------------------------


def _score_reap(ctx: MethodContext) -> ScoringResult:
    from crucible.methods.observer import compute_reap_scores, observe

    print("  Observing expert activations...")
    result = observe(ctx.model, ctx.dataloader, ctx.attrs)
    return ScoringResult(
        scores=compute_reap_scores(result),
        observation=result,
    )


def _score_pathfinder(ctx: MethodContext) -> ScoringResult:
    from crucible.methods.pathfinder import pathfinder_score

    scores, per_layer_keep = pathfinder_score(
        ctx.model, ctx.dataloader, ctx.attrs, target_ratio=ctx.ratio,
    )
    keep_range = f"{min(per_layer_keep)}-{max(per_layer_keep)}"
    print(f"  Per-layer experts: {keep_range} (non-uniform)")
    return ScoringResult(scores=scores, per_layer_keep=per_layer_keep)


def _score_task_aware(ctx: MethodContext) -> ScoringResult:
    from crucible.data import build_calibration_dataloader
    from crucible.methods.task_aware import compute_task_aware_scores

    opts = ctx.options
    print("  Building general-domain dataloader for contrast...")
    # Use Wikipedia for contrast — heavy on factual knowledge retrieval,
    # light on compositional reasoning. We want to identify experts that do
    # fact recall (expendable) vs language reasoning (shared with code, must
    # keep).
    general_dl = build_calibration_dataloader(
        ctx.tokenizer,
        datasets=["wikimedia/wikipedia"],
        num_samples=opts["samples"],
        max_seq_length=opts["max_seq_length"],
        batch_size=opts["batch_size"],
        seed=opts["seed"] + 1,
    )
    scores = compute_task_aware_scores(
        ctx.model, ctx.dataloader, general_dl, ctx.attrs,
        alpha=opts.get("task_alpha", 1.0),
    )
    return ScoringResult(scores=scores)


def _task_aware_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-alpha",
        type=float,
        default=1.0,
        help="Task-awareness strength for --scoring task-aware (default: 1.0)",
    )


def _score_adaptive(ctx: MethodContext) -> ScoringResult:
    from crucible.methods.adaptive import compute_adaptive_keep
    from crucible.methods.observer import compute_reap_scores, observe

    print("  Observing expert activations (REAP base)...")
    result = observe(ctx.model, ctx.dataloader, ctx.attrs)
    scores = compute_reap_scores(result)

    print("  Computing adaptive per-layer allocation...")
    per_layer_keep = compute_adaptive_keep(
        scores, target_ratio=ctx.ratio, min_keep=ctx.top_k,
    )
    return ScoringResult(
        scores=scores, per_layer_keep=per_layer_keep, observation=result,
    )


register_scorer(Scorer(
    name="reap",
    summary="Router gate-value x activation norm (the validated default)",
    score=_score_reap,
))
register_scorer(Scorer(
    name="pathfinder",
    summary="Cross-layer path analysis, plans a non-uniform per-layer budget",
    score=_score_pathfinder,
))
register_scorer(Scorer(
    name="task-aware",
    summary="REAP contrasted against a general-domain pass to find specialists",
    score=_score_task_aware,
    add_arguments=_task_aware_arguments,
))
register_scorer(Scorer(
    name="adaptive",
    summary="REAP scores plus a non-uniform per-layer expert allocation",
    score=_score_adaptive,
))


# ---------------------------------------------------------------------------
# Compression methods
# ---------------------------------------------------------------------------


def _compress_reap(ctx: MethodContext) -> MethodResult:
    from crucible.methods.reap import prune

    if ctx.scores is None:
        raise ValueError("REAP needs expert scores — run a scorer first")

    print("  Pruning experts...")
    # Non-uniform per-layer counts when the scorer planned them, else uniform.
    keep_counts = ctx.per_layer_keep if ctx.per_layer_keep is not None else ctx.num_to_keep
    result = prune(ctx.model, ctx.scores, ctx.attrs, keep_counts)

    if result.pad_strategy == "cloned":
        print(
            "  NOTE: padded layers duplicate a kept expert (this router has no "
            "additive bias to mask padding out of routing)"
        )

    return MethodResult(info={
        "experts_kept": result.experts_kept,
        "experts_removed": result.experts_removed,
        "per_layer_keep": ctx.per_layer_keep,  # None for uniform
        # Recorded because it changes what the artifact is: "masked" padding is
        # inert, "cloned" padding is reachable and perturbs routing.
        "pad_strategy": result.pad_strategy,
        "grouped_routing": (
            None if result.grouped_routing is None
            else vars(result.grouped_routing)
        ),
    })


def _compress_ream(ctx: MethodContext) -> MethodResult:
    from crucible.methods.ream import merge

    opts = ctx.options
    sequential = not opts.get("no_sequential", False)
    group_size = opts.get("group_size", 16)
    mode = "sequential" if sequential else "one-shot"
    print(f"  Merging experts ({mode}, group_size={group_size})...")

    result = merge(
        ctx.model,
        ctx.dataloader,
        ctx.attrs,
        ctx.num_to_keep,
        group_size=group_size,
        sequential=sequential,
        alignment=not opts.get("no_alignment", False),
        magnitude_correction=not opts.get("no_magnitude_correction", False),
        expert_similarity=not opts.get("no_expert_similarity", False),
        merge_float32=opts.get("merge_float32", False),
    )

    return MethodResult(info={
        "groups": [
            {str(k): v for k, v in layer_groups.items()}
            for layer_groups in result.groups
        ],
    })


def _ream_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group-size", type=int, default=16, help="REAM group size cap (default: 16)"
    )
    parser.add_argument(
        "--no-sequential",
        action="store_true",
        help="Disable sequential layer-by-layer processing (faster, lower quality)",
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Disable Hungarian neuron alignment before merging",
    )
    parser.add_argument(
        "--no-magnitude-correction",
        action="store_true",
        help="Disable post-merge magnitude correction",
    )
    parser.add_argument(
        "--no-expert-similarity",
        action="store_true",
        help="Use only gate similarity for grouping (skip expert output similarity)",
    )
    parser.add_argument(
        "--merge-float32",
        action="store_true",
        help="Accumulate merge weights in float32 instead of model dtype",
    )


register_method(Method(
    name="reap",
    summary="Router-weighted Expert Activation Pruning, the validated path",
    compress=_compress_reap,
    uses_scores=True,
))
register_method(Method(
    name="ream",
    summary="merge similar experts into high-saliency centroids; experimental, "
            "known-broken on Gemma 4's SiLU experts",
    compress=_compress_ream,
    # REAM observes and scores per layer as it merges, so a separate scoring
    # pass would be a wasted calibration run.
    uses_scores=False,
    add_arguments=_ream_arguments,
))
