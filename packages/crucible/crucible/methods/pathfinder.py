"""MoE Pathfinder: trajectory-driven expert scoring.

Models the MoE as a directed acyclic graph where experts are nodes and
inter-layer connections are edges. Finds globally optimal expert subsets
via dynamic programming on the top-m highest-weight paths.

Unlike per-layer REAP scoring, Pathfinder captures cross-layer expert
circuits — chains of experts that co-activate and depend on each other.

Reference: arxiv.org/abs/2512.18425

Usage:
    scores, per_layer_keep = pathfinder_score(
        model, dataloader, attrs, target_ratio=0.375
    )
    # scores can be used with ream.merge() or reap.prune()
    # per_layer_keep gives non-uniform per-layer expert counts
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crucible.methods.observer import (
    ExpertScore,
    _compute_expert_output,
    _find_layers,
    _get_config_value,
    _parse_router_output,
    _resolve_expert_activation,
    _resolve_path,
    _to_2d,
)
from crucible.types import ModelAttrs


def pathfinder_score(
    model: nn.Module,
    dataloader,
    attrs: ModelAttrs,
    target_ratio: float = 0.375,
    top_m_paths: int = 100,
) -> tuple[list[list[ExpertScore]], list[int]]:
    """Score experts using cross-layer path analysis.

    Builds a weighted computation graph across all MoE layers, finds
    the top-m highest-weight paths via DP, and scores each expert by
    how many important paths it appears in.

    Args:
        model: HuggingFace model (eval mode, on device).
        dataloader: calibration data.
        attrs: model architecture mapping.
        target_ratio: fraction of experts to remove (for per-layer planning).
        top_m_paths: number of top paths to find per calibration batch.

    Returns:
        (scores, per_layer_keep): scores in same format as compute_reap_scores,
        plus a per-layer list of how many experts to keep.
    """
    layers = _find_layers(model)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    device = next(model.parameters()).device

    moe_layer_indices = []
    for i, layer in enumerate(layers):
        try:
            _resolve_path(layer, attrs.router)
            _resolve_path(layer, attrs.experts)
            moe_layer_indices.append(i)
        except AttributeError:
            continue

    num_moe_layers = len(moe_layer_indices)

    # Collect per-layer statistics via hooks
    activation_norms, routing_probs, reconstruction_errors = _collect_stats(
        model, dataloader, layers, moe_layer_indices, attrs,
        num_experts, top_k, device,
    )

    # Build transition intensities: t[l][i][j] = a_i^l * r_j^{l+1}
    # activation_norms[l]: [num_experts] — average L2 norm of expert outputs
    # routing_probs[l]: [num_experts] — average routing probability

    # Expert importance from reconstruction error
    expert_importance = []
    for li in range(num_moe_layers):
        # Convert losses to importance via softmax(-loss)
        losses = reconstruction_errors[li]  # [num_experts]
        importance = F.softmax(-losses, dim=0)
        expert_importance.append(importance)

    # Path appearance count: how many top-m paths include each expert
    path_counts = torch.zeros(num_moe_layers, num_experts)

    # Run DP path finding per calibration batch's statistics
    # (we aggregate statistics across batches, so run DP once)
    paths = _find_top_paths(
        activation_norms, routing_probs, expert_importance,
        num_moe_layers, num_experts, top_m_paths,
    )

    for path, _weight in paths:
        for layer_pos, expert_idx in enumerate(path):
            path_counts[layer_pos, expert_idx] += 1

    # Convert path counts to scores
    all_scores = []
    for li in range(num_moe_layers):
        layer_idx = moe_layer_indices[li]
        layer_scores = []
        for e in range(num_experts):
            score = path_counts[li, e].item()
            layer_scores.append(
                ExpertScore(
                    layer_idx=layer_idx,
                    expert_idx=e,
                    score=score,
                    frequency=routing_probs[li][e].item(),
                    activation_norm=activation_norms[li][e].item(),
                    router_weight=routing_probs[li][e].item(),
                )
            )
        all_scores.append(layer_scores)

    # Compute non-uniform per-layer keep counts from path counts
    per_layer_keep = _compute_per_layer_keep(
        path_counts, num_experts, target_ratio, top_k,
    )

    return all_scores, per_layer_keep


def _collect_stats(
    model: nn.Module,
    dataloader,
    layers: list[nn.Module],
    moe_layer_indices: list[int],
    attrs: ModelAttrs,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Collect activation norms, routing probs, and reconstruction errors."""
    num_moe = len(moe_layer_indices)

    # Resolve the expert activation from the model, once, exactly as observe()
    # does — a scorer that assumed a different activation than the REAP scorer
    # would rank the same model on a different quantity.
    first_experts = _resolve_path(layers[moe_layer_indices[0]], attrs.experts)
    expert_act = _resolve_expert_activation(
        first_experts if attrs.expert_storage == "tensor3d" else first_experts[0],
        attrs,
        model.config,
    )

    # Accumulators
    norm_sum = [torch.zeros(num_experts, device=device) for _ in range(num_moe)]
    norm_count = [torch.zeros(num_experts, device=device) for _ in range(num_moe)]
    route_sum = [torch.zeros(num_experts, device=device) for _ in range(num_moe)]
    route_count = [0] * num_moe
    recon_error = [torch.zeros(num_experts, device=device) for _ in range(num_moe)]
    recon_count = [0] * num_moe

    # Hook routers
    captured: dict[int, dict] = {}
    hooks = []

    for si, layer_idx in enumerate(moe_layer_indices):
        router = _resolve_path(layers[layer_idx], attrs.router)
        idx = si

        def _pre(idx=idx):
            def hook(module, args):
                h = args[0] if isinstance(args, tuple) else args
                if isinstance(h, torch.Tensor):
                    captured[idx] = {"hidden": h}
            return hook

        def _post(idx=idx):
            def hook(module, args, output):
                captured.setdefault(idx, {})["router_output"] = output
            return hook

        hooks.append(router.register_forward_pre_hook(_pre()))
        hooks.append(router.register_forward_hook(_post()))

    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                model(input_ids=input_ids, attention_mask=attn_mask)

                for si, layer_idx in enumerate(moe_layer_indices):
                    if si not in captured:
                        continue

                    hidden = _to_2d(captured[si]["hidden"])
                    router_output = captured[si]["router_output"]
                    top_k_w, top_k_i = _parse_router_output(router_output, top_k)
                    top_k_w = _to_2d(top_k_w)
                    top_k_i = _to_2d(top_k_i)

                    experts_module = _resolve_path(
                        layers[layer_idx], attrs.experts
                    )

                    # Routing probabilities (average softmax scores)
                    if hasattr(router_output, '__len__') and len(router_output) >= 3:
                        full_probs = _to_2d(router_output[0])
                    else:
                        logits = _to_2d(
                            router_output if isinstance(router_output, torch.Tensor)
                            else router_output[0]
                        )
                        full_probs = F.softmax(logits, dim=-1)
                    route_sum[si] += full_probs.sum(dim=0)
                    route_count[si] += full_probs.shape[0]

                    # Activation norms + reconstruction error per expert
                    # (subsample for speed — use first 256 tokens per batch)
                    sample_h = hidden[:256]
                    active_experts = top_k_i[:256].unique()

                    # Full MoE output for reconstruction comparison
                    # Approximate: sum of gated expert outputs for selected experts
                    full_output = torch.zeros_like(sample_h)
                    for k in range(min(top_k, top_k_i.shape[1])):
                        for eidx in top_k_i[:256, k].unique():
                            mask = top_k_i[:256, k] == eidx
                            if not mask.any():
                                continue
                            out = _compute_expert_output(
                                sample_h[mask], experts_module, attrs,
                                eidx.item(), expert_act,
                            )
                            full_output[mask] += top_k_w[:256, k][mask].unsqueeze(-1) * out

                    for eidx in active_experts:
                        ei = eidx.item()
                        mask = (top_k_i[:256] == eidx).any(dim=-1)
                        if not mask.any():
                            continue

                        expert_out = _compute_expert_output(
                            sample_h[mask], experts_module, attrs, ei, expert_act
                        )

                        # Activation norm
                        norms = expert_out.float().norm(dim=-1)
                        norm_sum[si][ei] += norms.sum()
                        norm_count[si][ei] += norms.shape[0]

                        # Reconstruction error: how well does this single
                        # expert approximate the full MoE output?
                        single_approx = expert_out
                        target = full_output[mask]
                        error = (target - single_approx).pow(2).sum(dim=-1).mean()
                        recon_error[si][ei] += error
                        recon_count[si] += 1

                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    # Normalize
    activation_norms = []
    routing_probs = []
    reconstruction_errors = []

    for si in range(num_moe):
        valid = norm_count[si] > 0
        norms = torch.zeros(num_experts, device=device)
        norms[valid] = norm_sum[si][valid] / norm_count[si][valid]
        activation_norms.append(norms)

        if route_count[si] > 0:
            routing_probs.append(route_sum[si] / route_count[si])
        else:
            routing_probs.append(torch.ones(num_experts, device=device) / num_experts)

        if recon_count[si] > 0:
            reconstruction_errors.append(recon_error[si] / recon_count[si])
        else:
            reconstruction_errors.append(torch.zeros(num_experts, device=device))

    return activation_norms, routing_probs, reconstruction_errors


def _find_top_paths(
    activation_norms: list[torch.Tensor],
    routing_probs: list[torch.Tensor],
    expert_importance: list[torch.Tensor],
    num_layers: int,
    num_experts: int,
    top_m: int,
) -> list[tuple[list[int], float]]:
    """Find top-m highest-weight paths through the expert graph via DP.

    Path weight (in log space):
        log w = sum_l [log t(i_l, i_{l+1}) + log e(i_l)]

    Where:
        t(i, j) = activation_norm[l][i] * routing_prob[l+1][j]  (transition)
        e(i) = expert_importance[l][i]  (node weight)
    """
    # Convert to log space (add small epsilon for numerical stability)
    eps = 1e-10
    log_norms = [torch.log(a + eps).cpu() for a in activation_norms]
    log_routes = [torch.log(r + eps).cpu() for r in routing_probs]
    log_importance = [torch.log(e + eps).cpu() for e in expert_importance]

    # DP: for each expert at each layer, maintain top-m prefix paths
    # State: priority queue of (neg_log_weight, path) per (layer, expert)

    # Initialize layer 0
    # Each expert starts with its own importance
    current_best: list[list[list[tuple[float, list[int]]]]] = []

    layer0 = []
    for e in range(num_experts):
        w = log_importance[0][e].item()
        layer0.append([(w, [e])])
    current_best.append(layer0)

    # Forward pass through layers
    for li in range(1, num_layers):
        layer_l = [[] for _ in range(num_experts)]

        for j in range(num_experts):
            candidates = []
            for i in range(num_experts):
                # Transition: activation[li-1][i] * routing[li][j]
                trans = log_norms[li - 1][i].item() + log_routes[li][j].item()

                for prev_weight, prev_path in current_best[li - 1][i]:
                    new_weight = prev_weight + trans + log_importance[li][j].item()
                    candidates.append((new_weight, prev_path + [j]))

            # Keep top-m by weight (highest = best)
            candidates.sort(key=lambda x: x[0], reverse=True)
            layer_l[j] = candidates[:top_m]

        current_best.append(layer_l)

    # Collect all paths at the final layer
    all_final = []
    for e in range(num_experts):
        for weight, path in current_best[num_layers - 1][e]:
            all_final.append((path, weight))

    # Return top-m overall
    all_final.sort(key=lambda x: x[1], reverse=True)
    return all_final[:top_m]


def _compute_per_layer_keep(
    path_counts: torch.Tensor,
    num_experts: int,
    target_ratio: float,
    min_keep: int,
) -> list[int]:
    """Compute per-layer expert counts from path appearance frequency.

    Experts that appear in many top paths are important. Layers where
    path counts are concentrated need fewer experts removed (they're
    more sensitive). Layers where counts are spread need more experts.
    """
    num_layers = path_counts.shape[0]
    target_keep = int(num_experts * (1 - target_ratio))

    per_layer = []
    for li in range(num_layers):
        counts = path_counts[li]
        active = (counts > 0).sum().item()
        keep = max(min_keep, min(num_experts, max(target_keep, active)))
        per_layer.append(keep)

    # Adjust to match overall target
    total_target = target_keep * num_layers
    total_actual = sum(per_layer)
    diff = total_actual - total_target

    if diff > 0:
        excess = [(per_layer[li] - min_keep, li) for li in range(num_layers)]
        excess.sort(reverse=True)
        for _, idx in excess:
            if diff <= 0:
                break
            can_remove = per_layer[idx] - min_keep
            remove = min(can_remove, diff)
            per_layer[idx] -= remove
            diff -= remove

    return per_layer
