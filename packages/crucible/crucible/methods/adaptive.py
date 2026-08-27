"""Adaptive per-layer expert allocation for REAP pruning.

Instead of removing the same number of experts from every layer, allocates
the pruning budget where it hurts least. Uses relative marginal cost to
normalize across layers with different score scales.

The key insight: some layers have clear separation between important and
unimportant experts (safe to prune aggressively), while others have
nearly identical scores (dangerous to prune at all). Adaptive allocation
prunes more from well-separated layers and protects tight-distribution layers.

Algorithm: greedy relative marginal cost.
  1. Sort each layer's experts by score (ascending = cheapest to remove first).
  2. At each step, find the layer where removing one more expert has the
     lowest relative cost (score / layer median).
  3. Remove that expert. Repeat until budget is met.

This naturally handles:
  - Layers with different score scales (normalized by median)
  - Self-balancing: as a layer loses experts, the next removal gets more
    expensive, steering pruning toward other layers
  - Layers with clear junk experts (cost ~0) get pruned first

Usage:
    scores = compute_reap_scores(observation_result)
    per_layer_keep = compute_adaptive_keep(scores, target_ratio=0.375)
    prune_result = prune(model, scores, attrs, per_layer_keep)
"""

from __future__ import annotations

from crucible.types import ExpertScore


def compute_adaptive_keep(
    scores: list[list[ExpertScore]],
    target_ratio: float = 0.375,
    min_keep: int = 8,
) -> list[int]:
    """Compute non-uniform per-layer expert keep counts.

    Args:
        scores: per-layer expert scores from compute_reap_scores.
        target_ratio: fraction of experts to remove overall (same total as
            uniform pruning, just distributed differently).
        min_keep: minimum experts to keep per layer (must be >= top_k).

    Returns:
        List of per-layer keep counts. Total removed matches uniform pruning.
    """
    num_layers = len(scores)
    num_experts = len(scores[0])
    total_to_remove = int(num_experts * target_ratio) * num_layers

    # Sort each layer's scores ascending (weakest first = cheapest to remove)
    sorted_per_layer = []
    for layer in scores:
        sorted_per_layer.append(sorted(s.score for s in layer))

    # Layer medians for normalization — robust to outliers unlike mean
    medians = []
    for li in range(num_layers):
        medians.append(sorted_per_layer[li][num_experts // 2])

    # Greedy allocation: remove the globally cheapest expert at each step
    removed = [0] * num_layers
    max_removable = num_experts - min_keep

    for _ in range(total_to_remove):
        best_layer = -1
        best_cost = float("inf")

        for li in range(num_layers):
            if removed[li] >= max_removable:
                continue
            absolute_cost = sorted_per_layer[li][removed[li]]
            relative_cost = absolute_cost / max(medians[li], 1e-10)
            if relative_cost < best_cost:
                best_cost = relative_cost
                best_layer = li

        if best_layer == -1:
            break  # all layers maxed out
        removed[best_layer] += 1

    per_layer_keep = [num_experts - r for r in removed]

    # Report
    uniform = num_experts - int(num_experts * target_ratio)
    _print_allocation_summary(per_layer_keep, uniform)

    return per_layer_keep


def _print_allocation_summary(per_layer_keep: list[int], uniform: int) -> None:
    """Print a concise summary of the adaptive allocation."""
    keep_min = min(per_layer_keep)
    keep_max = max(per_layer_keep)
    print(f"    Adaptive allocation: {keep_min}-{keep_max} experts/layer "
          f"(uniform would be {uniform})")

    deltas = [(li, k - uniform) for li, k in enumerate(per_layer_keep)]
    most_pruned = sorted(deltas, key=lambda x: x[1])[:3]
    least_pruned = sorted(deltas, key=lambda x: x[1], reverse=True)[:3]
    print(f"    Most pruned:  {', '.join(f'L{li}({d:+d})' for li, d in most_pruned)}")
    print(f"    Least pruned: {', '.join(f'L{li}({d:+d})' for li, d in least_pruned)}")
