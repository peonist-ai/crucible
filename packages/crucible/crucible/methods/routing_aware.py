"""Routing-aware expert scoring for MoE compression.

Adjusts base saliency scores by accounting for routing disruption cost.
When pruning an expert, measures how well surviving experts can absorb
its traffic. Experts whose tokens have good alternatives get lower
disruption (safe to prune); experts whose tokens have no good fallback
get higher disruption (costly to prune).

Works as a modifier on top of any base scoring (REAP, task-aware, etc.).

Usage:
    base_scores = compute_reap_scores(observe(model, dataloader, attrs))
    adjusted = adjust_for_routing_disruption(
        model, dataloader, attrs, base_scores,
        num_to_keep=80, beta=1.0,
    )
    # Use adjusted scores with reap.prune()
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crucible.methods.observer import (
    ExpertScore,
    _extract_full_logits,
    _find_layers,
    _get_config_value,
    _parse_router_output,
    _resolve_path,
    _to_2d,
)
from crucible.types import ModelAttrs


def adjust_for_routing_disruption(
    model: nn.Module,
    dataloader,
    attrs: ModelAttrs,
    base_scores: list[list[ExpertScore]],
    *,
    num_to_keep: int,
    beta: float = 1.0,
) -> list[list[ExpertScore]]:
    """Adjust expert scores by routing disruption cost.

    Runs a forward pass to collect per-token routing data, then
    boosts experts whose pruning would cause high routing disruption
    (tokens have no good alternative among survivors).

    Args:
        model: HuggingFace model (eval mode).
        dataloader: calibration data.
        attrs: model architecture mapping.
        base_scores: per-layer expert scores from any scoring method.
        num_to_keep: target number of experts to keep per layer.
        beta: weight of routing disruption boost.
            0 = no adjustment (same as base scores).
            1.0 = moderate disruption protection.
            2.0+ = strongly protect hard-to-replace experts.

    Returns:
        Adjusted per-layer expert scores (same format, drop-in).
    """
    if beta == 0:
        return base_scores

    import time as _time

    layers = _find_layers(model)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    device = next(model.parameters()).device

    moe_layer_indices = _find_moe_layer_indices(layers, attrs)

    # Determine initial keep/prune sets per layer from base scores
    layer_keep_sets = []
    for layer_scores in base_scores:
        ranked = sorted(layer_scores, key=lambda s: s.score, reverse=True)
        keep_set = frozenset(s.expert_idx for s in ranked[:num_to_keep])
        layer_keep_sets.append(keep_set)

    print(
        f"    Computing routing disruption (beta={beta}, "
        f"{len(moe_layer_indices)} MoE layers)...",
        flush=True,
    )
    t0 = _time.time()

    all_disruption = _collect_disruption_all_layers(
        model, dataloader, layers, moe_layer_indices, attrs,
        num_experts, top_k, device, layer_keep_sets,
    )

    elapsed = _time.time() - t0
    # Summary stats
    total_disrupted = 0
    for layer_d in all_disruption:
        total_disrupted += sum(1 for v in layer_d.values() if v > 0.01)
    print(
        f"      done ({elapsed:.0f}s) — "
        f"{total_disrupted} experts with significant routing disruption",
        flush=True,
    )

    # Adjust scores per layer
    adjusted = []
    for layer_scores, disruption in zip(base_scores, all_disruption):
        adjusted.append(_adjust_layer_scores(layer_scores, disruption, beta))

    return adjusted


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_moe_layer_indices(
    layers: list[nn.Module], attrs: ModelAttrs
) -> list[int]:
    indices = []
    for i, layer in enumerate(layers):
        try:
            _resolve_path(layer, attrs.router)
            _resolve_path(layer, attrs.experts)
            indices.append(i)
        except AttributeError:
            continue
    if not indices:
        raise ValueError("No MoE layers found")
    return indices


def _collect_disruption_all_layers(
    model: nn.Module,
    dataloader,
    layers: list[nn.Module],
    moe_layer_indices: list[int],
    attrs: ModelAttrs,
    num_experts: int,
    top_k: int,
    device: torch.device,
    layer_keep_sets: list[frozenset[int]],
) -> list[dict[int, float]]:
    """Collect routing disruption for all MoE layers in a single pass."""

    # Per-layer accumulators
    disruption_sum = [
        torch.zeros(num_experts, dtype=torch.float64)
        for _ in moe_layer_indices
    ]
    disruption_count = [
        torch.zeros(num_experts, dtype=torch.int64)
        for _ in moe_layer_indices
    ]

    # Per-layer survive masks (precompute once)
    survive_masks = []
    for keep_set in layer_keep_sets:
        mask = torch.zeros(num_experts, dtype=torch.bool)
        for idx in keep_set:
            mask[idx] = True
        survive_masks.append(mask)

    # Hook into all routers
    captured: dict[int, dict] = {}
    hooks = []

    for stat_idx, layer_idx in enumerate(moe_layer_indices):
        router = _resolve_path(layers[layer_idx], attrs.router)
        si = stat_idx

        def _make_post(si=si):
            def hook(module, args, output):
                captured[si] = {"router_output": output}
            return hook

        hooks.append(router.register_forward_hook(_make_post()))

    model.eval()
    total_tokens = 0
    n_batches = len(dataloader)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(
                        f"      batch {batch_idx+1}/{n_batches}...",
                        end="\r", flush=True,
                    )
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                model(input_ids=input_ids, attention_mask=attn_mask)
                total_tokens += input_ids.numel()

                for stat_idx in range(len(moe_layer_indices)):
                    if stat_idx not in captured:
                        continue

                    router_output = captured[stat_idx]["router_output"]
                    top_k_w, top_k_i = _parse_router_output(
                        router_output, top_k
                    )
                    top_k_w = _to_2d(top_k_w)
                    top_k_i = _to_2d(top_k_i)

                    full_probs = _extract_full_logits(router_output)
                    if full_probs is None:
                        continue

                    _accumulate_disruption(
                        full_probs, top_k_w, top_k_i,
                        survive_masks[stat_idx], num_experts,
                        disruption_sum[stat_idx],
                        disruption_count[stat_idx],
                    )

                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    # Convert to per-expert disruption scores
    all_disruption = []
    for stat_idx in range(len(moe_layer_indices)):
        layer_disruption = {}
        for expert_idx in range(num_experts):
            count = disruption_count[stat_idx][expert_idx].item()
            if count > 0:
                layer_disruption[expert_idx] = (
                    disruption_sum[stat_idx][expert_idx].item() / count
                )
            else:
                layer_disruption[expert_idx] = 0.0
        all_disruption.append(layer_disruption)

    return all_disruption


def _accumulate_disruption(
    full_probs: torch.Tensor,      # [num_tokens, num_experts] CPU
    top_k_weights: torch.Tensor,   # [num_tokens, top_k] device
    top_k_indices: torch.Tensor,   # [num_tokens, top_k] device
    survive_mask: torch.Tensor,    # [num_experts] bool
    num_experts: int,
    disruption_sum: torch.Tensor,  # [num_experts] accumulator
    disruption_count: torch.Tensor,  # [num_experts] accumulator
) -> None:
    """Accumulate routing disruption for one batch.

    For each token routing to a prune-candidate expert e:
      disruption(e, t) = gate_weight(e,t) × (1 - best_surviving_prob(t) / prob(e,t))

    High value = token strongly prefers e with no good fallback.
    Low value = token has a good alternative among survivors.
    """
    top_k_w = top_k_weights.cpu().float()
    top_k_i = top_k_indices.cpu()
    probs = full_probs.float()

    # Prune candidates = experts NOT in survive set
    prune_candidates = [
        i for i in range(num_experts) if not survive_mask[i]
    ]

    for expert_idx in prune_candidates:
        # Tokens routing to this expert
        mask = top_k_i == expert_idx  # [num_tokens, top_k]
        token_mask = mask.any(dim=-1)  # [num_tokens]

        if not token_mask.any():
            continue

        n_affected = token_mask.sum().item()

        # This expert's gate weight on affected tokens
        gate = (top_k_w * mask.float()).sum(dim=-1)[token_mask]

        # This expert's router probability on affected tokens
        prob_e = probs[token_mask, expert_idx]

        # Best surviving expert's probability on affected tokens
        surviving_probs = probs[token_mask][:, survive_mask]
        best_surviving = surviving_probs.max(dim=-1).values

        # Disruption: how much worse is the best alternative?
        ratio = torch.where(
            prob_e > 1e-8,
            best_surviving / prob_e,
            torch.ones_like(prob_e),
        )
        token_disruption = gate * (1.0 - ratio).clamp(min=0)

        disruption_sum[expert_idx] += token_disruption.sum().double()
        disruption_count[expert_idx] += n_affected


def _adjust_layer_scores(
    layer_scores: list[ExpertScore],
    disruption: dict[int, float],
    beta: float,
) -> list[ExpertScore]:
    """Adjust layer scores: boost high-disruption experts (costly to prune)."""
    values = [v for v in disruption.values() if v > 0]
    if not values:
        return layer_scores

    max_disruption = max(values)
    if max_disruption < 1e-8:
        return layer_scores

    adjusted = []
    for s in layer_scores:
        d = disruption.get(s.expert_idx, 0.0)
        d_norm = d / max_disruption
        # boost > 1 means "this expert is hard to replace, keep it"
        boost = 1.0 + beta * d_norm

        adjusted.append(ExpertScore(
            layer_idx=s.layer_idx,
            expert_idx=s.expert_idx,
            score=s.score * boost,
            frequency=s.frequency,
            activation_norm=s.activation_norm,
            router_weight=s.router_weight,
        ))

    return adjusted
