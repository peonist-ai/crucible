"""Task-aware asymmetric expert compression (REAP-T).

Scores experts separately on task-specific vs. general calibration data,
then boosts task-specialist experts during compression. This enables more
aggressive overall compression while preserving task-critical capability.

The key insight: for a coding specialist, losing the "Renaissance art"
expert is fine; losing the "function signatures" expert is catastrophic.
Standard REAP/REAM treats them equally. REAP-T does not.

Usage:
    task_scores = compute_task_aware_scores(
        model, task_dataloader, general_dataloader, attrs
    )
    # Use task_scores with ream.merge() or reap.prune()
"""

from __future__ import annotations

import torch.nn as nn

from crucible.methods.observer import (
    ExpertScore,
    compute_reap_scores,
    observe,
)
from crucible.types import ModelAttrs


def compute_task_aware_scores(
    model: nn.Module,
    task_dataloader,
    general_dataloader,
    attrs: ModelAttrs,
    *,
    alpha: float = 1.0,
    store_router_logits: bool = False,
) -> list[list[ExpertScore]]:
    """Compute task-aware expert scores via dual observation.

    Runs two observation passes — one on task-specific data (e.g., code),
    one on general data (e.g., C4). Computes a task-specificity ratio per
    expert, then boosts task-specialists in the final score.

    Args:
        model: HuggingFace model (eval mode, on device).
        task_dataloader: calibration data for the target task (code, tool-use).
        general_dataloader: general-domain calibration data (C4, Wikipedia).
        attrs: model architecture mapping.
        alpha: asymmetry strength. 0 = standard REAP (no task bias).
            1.0 = moderate bias. 2.0+ = aggressive task protection.
        store_router_logits: pass through to observer for REAM compatibility.

    Returns:
        Per-layer expert scores with task-aware boosting applied.
        Same format as compute_reap_scores() — drop-in replacement.
    """
    import time as _time

    # Observe on task-specific data
    print("    Pass 1/2: observing on task-specific data...", flush=True)
    t0 = _time.time()
    task_result = observe(
        model, task_dataloader, attrs,
        store_router_logits=store_router_logits,
    )
    task_scores = compute_reap_scores(task_result)
    print(f"      done ({_time.time() - t0:.0f}s)", flush=True)

    # Observe on general data
    print("    Pass 2/2: observing on general data...", flush=True)
    t0 = _time.time()
    general_result = observe(
        model, general_dataloader, attrs,
        store_router_logits=store_router_logits,
    )
    general_scores = compute_reap_scores(general_result)
    print(f"      done ({_time.time() - t0:.0f}s)", flush=True)

    # Compute task-aware scores
    boosted = []
    for layer_idx in range(len(task_scores)):
        layer_boosted = _boost_layer(
            task_scores[layer_idx],
            general_scores[layer_idx],
            alpha,
        )
        boosted.append(layer_boosted)

    return boosted


def compute_task_specificity(
    task_scores: list[list[ExpertScore]],
    general_scores: list[list[ExpertScore]],
) -> list[list[float]]:
    """Compute per-expert task-specificity ratios.

    Returns values in [0, 1]:
        → 1.0: pure task specialist (only activated by task data)
        → 0.5: generalist (equally activated by both)
        → 0.0: task-irrelevant (only activated by general data)

    Useful for analysis and visualization of expert roles.
    """
    ratios = []
    for layer_idx in range(len(task_scores)):
        layer_ratios = []
        for t, g in zip(task_scores[layer_idx], general_scores[layer_idx]):
            total = t.score + g.score
            if total > 0:
                layer_ratios.append(t.score / total)
            else:
                layer_ratios.append(0.5)
        ratios.append(layer_ratios)
    return ratios


def _boost_layer(
    task_layer: list[ExpertScore],
    general_layer: list[ExpertScore],
    alpha: float,
) -> list[ExpertScore]:
    """Apply task-aware score boosting to one layer.

    Final score = S_task × (1 + α × (R - 0.5))

    Where R = S_task / (S_task + S_general) is the task-specificity ratio.

    When α = 0: S_final = S_task (standard REAP on task data)
    When α > 0: task-specialists get boosted, generalists stay neutral,
                task-irrelevant experts get penalized
    """
    boosted = []
    for t, g in zip(task_layer, general_layer):
        total = t.score + g.score
        if total > 0:
            ratio = t.score / total
        else:
            ratio = 0.5

        # Don't boost/penalize experts with too few activations —
        # the ratio is unreliable. Treat them as neutral generalists.
        min_freq = 0.01  # must be activated on ≥1% of tokens
        if t.frequency < min_freq and g.frequency < min_freq:
            ratio = 0.5

        # Boost: (R - 0.5) ranges from -0.5 to +0.5
        # So the multiplier ranges from (1 - α/2) to (1 + α/2)
        boost = 1.0 + alpha * (ratio - 0.5)
        final_score = t.score * boost

        boosted.append(
            ExpertScore(
                layer_idx=t.layer_idx,
                expert_idx=t.expert_idx,
                score=final_score,
                frequency=t.frequency,
                activation_norm=t.activation_norm,
                router_weight=t.router_weight,
            )
        )

    return boosted
