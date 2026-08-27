"""REAM: Router-weighted Expert Activation Merging.

Protects high-saliency experts as centroids, merges similar neighbors
into them via saliency-weighted averaging. Sequential layer-by-layer
processing with activation recomputation for quality preservation.

Reference: arxiv.org/abs/2604.04356

Usage:
    result = merge(model, dataloader, attrs, num_experts_to_keep=80)
    # model is now modified in-place with merged + pruned experts
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from crucible.methods.assignment import linear_sum_assignment
from crucible.methods.observer import (
    ExpertScore,
    LayerStats,
    _accumulate_layer,
    _extract_full_logits,
    _find_layers,
    _get_config_value,
    _init_layer_stats,
    _parse_router_output,
    _resolve_path,
    _to_2d,
)
from crucible.methods.reap import (
    _prune_experts,
    _prune_router,
    _update_config,
)
from crucible.types import ModelAttrs


@dataclass
class MergeResult:
    """Result of a REAM merge operation."""

    # Per-layer: centroid expert_idx -> list of merged member expert_idxs
    groups: list[dict[int, list[int]]]
    original_num_experts: int
    remaining_num_experts: int
    moe_layer_indices: list[int]


def merge(
    model: nn.Module,
    dataloader,
    attrs: ModelAttrs,
    num_experts_to_keep: int,
    *,
    group_size: int = 16,
    sequential: bool = True,
    alignment: bool = True,
    magnitude_correction: bool = True,
    expert_similarity: bool = True,
    merge_float32: bool = False,
) -> MergeResult:
    """Merge and prune experts using REAM.

    For each MoE layer:
      1. Score experts via REAP saliency
      2. Select top-scoring as centroids
      3. Assign non-centroids to most similar centroid (capped by group_size)
      4. Merge members into centroids via saliency-weighted averaging
      5. Prune non-centroid experts and update router

    In sequential mode, activations are recomputed after each layer's merge
    so subsequent layers see post-merge inputs. This roughly doubles compute
    but significantly improves quality (~8.7pt in ablation).

    Args:
        model: HuggingFace model (modified in-place).
        dataloader: calibration data, yields dicts with 'input_ids'.
        attrs: model architecture mapping.
        num_experts_to_keep: experts to retain per layer after merge+prune.
        group_size: max non-centroids a single centroid can absorb.
        sequential: recompute activations after each layer (recommended).

    Returns:
        MergeResult with per-layer group assignments.
    """
    layers = _find_layers(model)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    device = next(model.parameters()).device

    if num_experts_to_keep >= num_experts:
        raise ValueError(
            f"num_experts_to_keep ({num_experts_to_keep}) >= "
            f"num_experts ({num_experts})"
        )
    if num_experts_to_keep < top_k:
        raise ValueError(
            f"num_experts_to_keep ({num_experts_to_keep}) < top_k ({top_k})"
        )

    if not all([alignment, magnitude_correction, expert_similarity]) or merge_float32:
        print(f"    Merge config: alignment={alignment}, mag_correction={magnitude_correction}, "
              f"expert_sim={expert_similarity}, float32={merge_float32}")

    moe_layer_indices = _find_moe_layer_indices(layers, attrs)

    all_groups = []

    if sequential:
        import time as _time

        total_layers = len(moe_layer_indices)
        t_start = _time.time()

        for step, layer_idx in enumerate(moe_layer_indices):
            layer = layers[layer_idx]
            layer_num_experts = _count_layer_experts(layer, attrs)

            t_layer = _time.time()
            print(
                f"    Layer {step + 1}/{total_layers} "
                f"(model layer {layer_idx}, "
                f"{layer_num_experts} experts)...",
                flush=True,
            )

            obs = _observe_single_layer(
                model, dataloader, layer, attrs,
                layer_num_experts, top_k, device,
            )

            groups = _plan_and_merge_layer(
                layer, obs, attrs,
                num_experts_to_keep, group_size,
                magnitude_correction=magnitude_correction,
                alignment=alignment,
                expert_similarity=expert_similarity,
                merge_float32=merge_float32,
            )
            all_groups.append(groups)

            elapsed = _time.time() - t_layer
            total_elapsed = _time.time() - t_start
            merged_count = sum(len(v) for v in groups.values())
            eta = (total_elapsed / (step + 1)) * (total_layers - step - 1)
            print(
                f"      {elapsed:.0f}s — "
                f"merged {merged_count} experts into "
                f"{len(groups)} centroids — "
                f"ETA {eta / 60:.0f}m",
                flush=True,
            )
    else:
        # One-shot: observe all layers on the original model, then merge all
        all_layer_data = _observe_all_layers(
            model, dataloader, layers, moe_layer_indices, attrs,
            num_experts, top_k, device,
        )
        for layer_idx, obs in zip(moe_layer_indices, all_layer_data):
            layer = layers[layer_idx]
            groups = _plan_and_merge_layer(
                layer, obs, attrs,
                num_experts_to_keep, group_size,
                magnitude_correction=magnitude_correction,
                alignment=alignment,
                expert_similarity=expert_similarity,
                merge_float32=merge_float32,
            )
            all_groups.append(groups)

    _update_config(model.config, attrs.num_experts_key, num_experts_to_keep)

    return MergeResult(
        groups=all_groups,
        original_num_experts=num_experts,
        remaining_num_experts=num_experts_to_keep,
        moe_layer_indices=moe_layer_indices,
    )


# ---------------------------------------------------------------------------
# Layer-level merge pipeline
# ---------------------------------------------------------------------------


def _plan_and_merge_layer(
    layer: nn.Module,
    obs: _LayerObservation,
    attrs: ModelAttrs,
    num_to_keep: int,
    group_size: int,
    *,
    magnitude_correction: bool = True,
    alignment: bool = True,
    expert_similarity: bool = True,
    merge_float32: bool = False,
) -> dict[int, list[int]]:
    """Plan groups, merge weights, then prune a single layer. Returns groups."""
    experts_module = _resolve_path(layer, attrs.experts)
    num_experts = len(obs.scores)

    # Combined similarity: gate routing + expert output
    hidden_on_device = None
    topk_w_on_device = None
    topk_i_on_device = None
    device = next(
        (p.device for p in experts_module.parameters()), torch.device("cpu")
    )
    if obs.hidden_sample is not None:
        hidden_on_device = obs.hidden_sample.to(device)
        topk_w_on_device = obs.topk_weights_sample.to(device)
        topk_i_on_device = obs.topk_indices_sample.to(device)

    similarity = _compute_combined_similarity(
        obs.router_logits,
        hidden_on_device if expert_similarity else None,
        topk_w_on_device if expert_similarity else None,
        topk_i_on_device if expert_similarity else None,
        experts_module if expert_similarity else None,
        attrs if expert_similarity else None,
        num_experts,
    )
    groups = _assign_groups(obs.scores, similarity, num_to_keep, group_size)

    saliency = {s.expert_idx: s.score for s in obs.scores}

    # Compute pre-merge output norms for magnitude correction
    pre_merge_norms = {}
    if magnitude_correction:
        for s in obs.scores:
            if s.activation_norm > 0:
                pre_merge_norms[s.expert_idx] = s.activation_norm

    # Merge member weights into centroids
    for centroid_idx, member_indices in groups.items():
        _merge_group_weights(
            experts_module, centroid_idx, member_indices, saliency, attrs,
            alignment=alignment, merge_float32=merge_float32,
        )

    # Magnitude correction: rescale merged centroids to preserve
    # expected output magnitude
    if magnitude_correction and pre_merge_norms and hidden_on_device is not None:
        from crucible.methods.observer import _compute_expert_output

        for centroid_idx, member_indices in groups.items():
            if not member_indices:
                continue
            if centroid_idx not in pre_merge_norms:
                continue

            # Expected norm = weighted average of group's pre-merge norms
            group_indices = [centroid_idx] + list(member_indices)
            group_norms = [
                pre_merge_norms.get(i, 0.0) for i in group_indices
            ]
            group_saliency = [saliency.get(i, 0.0) for i in group_indices]
            total_sal = sum(group_saliency)
            if total_sal == 0:
                continue
            expected_norm = sum(
                s * n for s, n in zip(group_saliency, group_norms)
            ) / total_sal

            # Compute actual post-merge output norm on a subsample
            sample_h = hidden_on_device[:256]
            post_out = _compute_expert_output(
                sample_h, experts_module, attrs, centroid_idx
            )
            actual_norm = post_out.float().norm(dim=-1).mean().item()

            if actual_norm > 1e-8:
                scale = expected_norm / actual_norm
                # Clamp to prevent extreme rescaling
                scale = max(0.5, min(2.0, scale))
                _rescale_expert(experts_module, centroid_idx, scale, attrs)

    # Prune non-centroid experts and router rows
    keep_indices = sorted(groups.keys())
    _prune_experts(layer, keep_indices, attrs)
    _prune_router(layer, keep_indices, attrs)

    return groups


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _compute_gate_similarity(
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Pairwise cosine similarity of expert routing patterns.

    Each expert's "pattern" is its logit/probability across all calibration
    tokens. Experts routed by similar tokens get high similarity.

    Args:
        router_logits: [num_tokens, num_experts]

    Returns:
        [num_experts, num_experts] similarity matrix in [-1, 1].
    """
    patterns = router_logits.float().T  # [num_experts, num_tokens]
    patterns = F.normalize(patterns, dim=1)
    return patterns @ patterns.T


def _compute_expert_output_similarity(
    hidden_states: torch.Tensor,
    top_k_weights: torch.Tensor,
    top_k_indices: torch.Tensor,
    experts_module: nn.Module,
    attrs: ModelAttrs,
    num_experts: int,
    max_tokens: int = 2048,
) -> torch.Tensor:
    """Pairwise cosine similarity of gated expert outputs.

    For each pair of experts (i, j), computes similarity of their
    gate-weighted outputs across tokens where either is active.
    This captures functional similarity — do they compute the same thing?

    Args:
        hidden_states: [num_tokens, hidden] input to experts.
        top_k_weights: [num_tokens, top_k] gate values.
        top_k_indices: [num_tokens, top_k] selected expert indices.
        experts_module: the experts module with weights.
        attrs: model architecture mapping.
        num_experts: total number of experts.
        max_tokens: subsample tokens if more than this (memory).

    Returns:
        [num_experts, num_experts] similarity matrix.
    """
    from crucible.methods.observer import _compute_expert_output

    n_tokens = hidden_states.shape[0]
    if n_tokens > max_tokens:
        perm = torch.randperm(n_tokens, device=hidden_states.device)[:max_tokens]
        hidden_states = hidden_states[perm]
        top_k_weights = top_k_weights[perm]
        top_k_indices = top_k_indices[perm]
        n_tokens = max_tokens

    hidden_dim = hidden_states.shape[1]

    # Compute mean gated output per expert: E[gate * output]
    # We accumulate sum and count, then normalize
    output_sum = torch.zeros(num_experts, hidden_dim, device=hidden_states.device)
    output_count = torch.zeros(num_experts, device=hidden_states.device)

    active_experts = top_k_indices.unique()

    for eidx_t in active_experts:
        eidx = eidx_t.item()
        mask = (top_k_indices == eidx)
        token_mask = mask.any(dim=-1)

        if not token_mask.any():
            continue

        gates = (top_k_weights * mask.float()).sum(dim=-1)[token_mask]
        expert_out = _compute_expert_output(
            hidden_states[token_mask], experts_module, attrs, eidx
        )
        # Gated output: gate * expert_output
        gated = gates.unsqueeze(-1) * expert_out.float()
        output_sum[eidx] = gated.sum(dim=0)
        output_count[eidx] = token_mask.sum().float()

    # Normalize to get mean gated output per expert
    valid = output_count > 0
    output_sum[valid] = output_sum[valid] / output_count[valid].unsqueeze(-1)

    # Cosine similarity of mean gated outputs
    output_sum = F.normalize(output_sum, dim=1)
    sim = output_sum @ output_sum.T

    # Zero out invalid entries
    invalid_mask = ~valid
    sim[invalid_mask, :] = 0
    sim[:, invalid_mask] = 0

    return sim


def _compute_combined_similarity(
    router_logits: torch.Tensor | None,
    hidden_states: torch.Tensor | None,
    top_k_weights: torch.Tensor | None,
    top_k_indices: torch.Tensor | None,
    experts_module: nn.Module | None,
    attrs: ModelAttrs | None,
    num_experts: int,
) -> torch.Tensor | None:
    """Compute combined REAM similarity: delta_gate + delta_expert.

    Falls back gracefully if components are missing.
    """
    gate_sim = None
    if router_logits is not None:
        gate_sim = _compute_gate_similarity(router_logits)

    expert_sim = None
    if (
        hidden_states is not None
        and top_k_weights is not None
        and top_k_indices is not None
        and experts_module is not None
        and attrs is not None
    ):
        expert_sim = _compute_expert_output_similarity(
            hidden_states, top_k_weights, top_k_indices,
            experts_module, attrs, num_experts,
        )

    if gate_sim is not None and expert_sim is not None:
        expert_sim = expert_sim.to(gate_sim.device)
        return gate_sim + expert_sim
    if gate_sim is not None:
        return gate_sim
    return expert_sim


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------


def _assign_groups(
    scores: list[ExpertScore],
    similarity: torch.Tensor | None,
    num_to_keep: int,
    group_size: int,
) -> dict[int, list[int]]:
    """Assign non-centroid experts to centroid groups.

    Centroids are the top-scoring experts. Non-centroids are assigned
    (weakest first) to the most similar centroid that hasn't hit the
    group cap. Unassigned experts are dropped (pure pruning).

    Returns:
        dict mapping centroid_idx -> list of member indices (NOT including
        the centroid itself).
    """
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    centroid_set = {s.expert_idx for s in ranked[:num_to_keep]}
    non_centroids = [s for s in ranked[num_to_keep:]]

    groups: dict[int, list[int]] = {idx: [] for idx in centroid_set}

    if not non_centroids:
        return groups

    # Assign weakest non-centroids first (greedy, as in paper)
    non_centroids.sort(key=lambda s: s.score)
    group_counts = {idx: 0 for idx in centroid_set}

    for nc in non_centroids:
        best_centroid = None
        best_sim = -float("inf")

        for c_idx in centroid_set:
            if group_counts[c_idx] >= group_size:
                continue
            if similarity is not None:
                sim = similarity[nc.expert_idx, c_idx].item()
            else:
                # Fallback: assign to centroid with highest saliency
                sim = next(
                    s.score for s in scores if s.expert_idx == c_idx
                )
            if sim > best_sim:
                best_sim = sim
                best_centroid = c_idx

        if best_centroid is not None:
            groups[best_centroid].append(nc.expert_idx)
            group_counts[best_centroid] += 1

    return groups


# ---------------------------------------------------------------------------
# Weight merging
# ---------------------------------------------------------------------------


def _rescale_expert(
    experts_module: nn.Module,
    expert_idx: int,
    scale: float,
    attrs: ModelAttrs,
) -> None:
    """Rescale an expert's down projection to adjust output magnitude."""
    if attrs.expert_storage == "tensor3d":
        down = getattr(experts_module, attrs.down_proj)
        down.data[expert_idx] *= scale
    else:
        expert = experts_module[expert_idx]
        down = getattr(expert, attrs.down_proj)
        down.weight.data *= scale
        if down.bias is not None:
            down.bias.data *= scale


def _merge_group_weights(
    experts_module: nn.Module,
    centroid_idx: int,
    member_indices: list[int],
    saliency: dict[int, float],
    attrs: ModelAttrs,
    *,
    alignment: bool = True,
    merge_float32: bool = False,
) -> None:
    """Merge member expert weights into centroid via saliency-weighted average."""
    if not member_indices:
        return

    all_indices = [centroid_idx] + list(member_indices)
    raw = torch.tensor(
        [saliency.get(i, 0.0) for i in all_indices], dtype=torch.float32
    )
    total = raw.sum()
    w = raw / total if total > 0 else torch.ones_like(raw) / len(raw)

    if attrs.expert_storage == "tensor3d":
        _merge_tensor3d(experts_module, centroid_idx, all_indices, w, attrs,
                         alignment=alignment, merge_float32=merge_float32)
    else:
        _merge_modulelist(experts_module, centroid_idx, all_indices, w, attrs,
                           alignment=alignment, merge_float32=merge_float32)


# ---------------------------------------------------------------------------
# Neuron alignment (Hungarian algorithm)
# ---------------------------------------------------------------------------


def _compute_alignment(
    centroid_weights: list[torch.Tensor],
    member_weights: list[torch.Tensor],
) -> torch.Tensor:
    """Find optimal neuron permutation to align member to centroid.

    Uses the Hungarian algorithm on an L2-distance cost matrix over
    the intermediate (neuron) dimension.

    Args:
        centroid_weights: list of weight matrices from the centroid expert.
            Rows correspond to neurons (gate_proj, up_proj rows; down_proj cols).
        member_weights: same structure from the member expert.

    Returns:
        permutation: LongTensor of length intermediate_size. permutation[i]
        gives the member neuron index that should map to centroid neuron i.
    """
    # Stack all neuron-indexed vectors for a combined cost matrix.
    # Each "neuron" contributes rows from gate/up and a column from down.
    c_vecs = torch.cat(centroid_weights, dim=-1).float()  # [intermediate, combined_dim]
    m_vecs = torch.cat(member_weights, dim=-1).float()

    # Pairwise L2 distance: cost[i, j] = ||c[i] - m[j]||^2
    # Expand for broadcasting: [I, 1, D] - [1, I, D] -> [I, I, D]
    cost = torch.cdist(c_vecs, m_vecs, p=2)

    # Hungarian algorithm (minimize total cost)
    _, col_ind = linear_sum_assignment(cost)

    return col_ind.to(c_vecs.device)


def _get_neuron_vectors_tensor3d(
    experts_module: nn.Module,
    expert_idx: int,
    attrs: ModelAttrs,
) -> list[torch.Tensor]:
    """Extract per-neuron weight vectors for a tensor3d expert.

    For fused gate_up [2*I, H], neurons are paired: rows i and i+I
    correspond to the same intermediate neuron.
    For down [H, I], neurons are columns.

    Returns list of [intermediate_size, *] tensors for cost matrix.
    """
    gate_up = getattr(experts_module, attrs.gate_proj).data[expert_idx]
    down = getattr(experts_module, attrs.down_proj).data[expert_idx]

    intermediate = down.shape[-1]  # down is [hidden, intermediate]

    if attrs.fused_gate_up:
        # gate_up is [2*intermediate, hidden]
        gate_rows = gate_up[:intermediate]   # [I, H]
        up_rows = gate_up[intermediate:]     # [I, H]
        down_cols = down.T                   # [I, H] (transpose [H, I])
        return [gate_rows, up_rows, down_cols]
    else:
        up = getattr(experts_module, attrs.up_proj).data[expert_idx]
        gate_rows = gate_up  # [I, H]
        up_rows = up         # [I, H]
        down_cols = down.T   # [I, H]
        return [gate_rows, up_rows, down_cols]


def _get_neuron_vectors_modulelist(
    experts_module: nn.ModuleList,
    expert_idx: int,
    attrs: ModelAttrs,
) -> list[torch.Tensor]:
    """Extract per-neuron weight vectors for a modulelist expert."""
    expert = experts_module[expert_idx]
    gate_w = getattr(expert, attrs.gate_proj).weight.data  # [I, H]
    down_w = getattr(expert, attrs.down_proj).weight.data  # [H, I]

    if attrs.fused_gate_up:
        intermediate = down_w.shape[-1]
        gate_rows = gate_w[:intermediate]
        up_rows = gate_w[intermediate:]
        return [gate_rows, up_rows, down_w.T]

    up_w = getattr(expert, attrs.up_proj).weight.data  # [I, H]
    return [gate_w, up_w, down_w.T]


def _apply_permutation_tensor3d(
    experts_module: nn.Module,
    expert_idx: int,
    perm: torch.Tensor,
    attrs: ModelAttrs,
) -> None:
    """Permute a tensor3d expert's neurons in-place."""
    gate_up = getattr(experts_module, attrs.gate_proj)
    down = getattr(experts_module, attrs.down_proj)

    intermediate = down.shape[-1]

    if attrs.fused_gate_up:
        # Permute gate rows and up rows in tandem
        old = gate_up.data[expert_idx].clone()
        gate_up.data[expert_idx][:intermediate] = old[:intermediate][perm]
        gate_up.data[expert_idx][intermediate:] = old[intermediate:][perm]
    else:
        up_proj = getattr(experts_module, attrs.up_proj)
        gate_up.data[expert_idx] = gate_up.data[expert_idx][perm]
        up_proj.data[expert_idx] = up_proj.data[expert_idx][perm]

    # down_proj: [hidden, intermediate] — permute columns
    down.data[expert_idx] = down.data[expert_idx][:, perm]


def _apply_permutation_modulelist(
    experts_module: nn.ModuleList,
    expert_idx: int,
    perm: torch.Tensor,
    attrs: ModelAttrs,
) -> None:
    """Permute a modulelist expert's neurons in-place."""
    expert = experts_module[expert_idx]
    gate = getattr(expert, attrs.gate_proj)
    down = getattr(expert, attrs.down_proj)

    if attrs.fused_gate_up:
        intermediate = down.weight.shape[-1]
        old = gate.weight.data.clone()
        gate.weight.data[:intermediate] = old[:intermediate][perm]
        gate.weight.data[intermediate:] = old[intermediate:][perm]
        if gate.bias is not None:
            old_b = gate.bias.data.clone()
            gate.bias.data[:intermediate] = old_b[:intermediate][perm]
            gate.bias.data[intermediate:] = old_b[intermediate:][perm]
    else:
        up = getattr(expert, attrs.up_proj)
        gate.weight.data = gate.weight.data[perm]
        up.weight.data = up.weight.data[perm]
        if gate.bias is not None:
            gate.bias.data = gate.bias.data[perm]
        if up.bias is not None:
            up.bias.data = up.bias.data[perm]

    # down: [hidden, intermediate] — permute columns
    down.weight.data = down.weight.data[:, perm]
    # down bias is over hidden dimension, not affected by neuron permutation


def _align_member_to_centroid(
    experts_module: nn.Module,
    centroid_idx: int,
    member_idx: int,
    attrs: ModelAttrs,
) -> None:
    """Align a member expert's neurons to the centroid and permute in-place."""
    if attrs.expert_storage == "tensor3d":
        c_vecs = _get_neuron_vectors_tensor3d(
            experts_module, centroid_idx, attrs
        )
        m_vecs = _get_neuron_vectors_tensor3d(
            experts_module, member_idx, attrs
        )
    else:
        c_vecs = _get_neuron_vectors_modulelist(
            experts_module, centroid_idx, attrs
        )
        m_vecs = _get_neuron_vectors_modulelist(
            experts_module, member_idx, attrs
        )

    perm = _compute_alignment(c_vecs, m_vecs)

    if attrs.expert_storage == "tensor3d":
        _apply_permutation_tensor3d(experts_module, member_idx, perm, attrs)
    else:
        _apply_permutation_modulelist(experts_module, member_idx, perm, attrs)


# ---------------------------------------------------------------------------
# Weight merging (with alignment)
# ---------------------------------------------------------------------------


def _merge_tensor3d(
    experts_module: nn.Module,
    centroid_idx: int,
    all_indices: list[int],
    weights: torch.Tensor,
    attrs: ModelAttrs,
    *,
    alignment: bool = True,
    merge_float32: bool = False,
) -> None:
    if alignment:
        for member_idx in all_indices[1:]:
            _align_member_to_centroid(experts_module, centroid_idx, member_idx, attrs)

    gate_up = getattr(experts_module, attrs.gate_proj)
    down = getattr(experts_module, attrs.down_proj)

    if merge_float32:
        merged_gu = sum(
            w.item() * gate_up.data[i].float() for w, i in zip(weights, all_indices)
        )
        merged_d = sum(
            w.item() * down.data[i].float() for w, i in zip(weights, all_indices)
        )
        gate_up.data[centroid_idx] = merged_gu.to(gate_up.data.dtype)
        down.data[centroid_idx] = merged_d.to(down.data.dtype)
    else:
        merged_gu = sum(
            w.item() * gate_up.data[i] for w, i in zip(weights, all_indices)
        )
        merged_d = sum(
            w.item() * down.data[i] for w, i in zip(weights, all_indices)
        )
        gate_up.data[centroid_idx] = merged_gu
        down.data[centroid_idx] = merged_d


def _merge_modulelist(
    experts_module: nn.ModuleList,
    centroid_idx: int,
    all_indices: list[int],
    weights: torch.Tensor,
    attrs: ModelAttrs,
    *,
    alignment: bool = True,
    merge_float32: bool = False,
) -> None:
    if alignment:
        for member_idx in all_indices[1:]:
            _align_member_to_centroid(experts_module, centroid_idx, member_idx, attrs)

    centroid = experts_module[centroid_idx]
    param_names = list(
        dict.fromkeys([attrs.gate_proj, attrs.up_proj, attrs.down_proj])
    )

    for pname in param_names:
        tensors = [
            getattr(experts_module[i], pname).weight.data
            for i in all_indices
        ]
        if merge_float32:
            merged = sum(w.item() * t.float() for w, t in zip(weights, tensors))
            getattr(centroid, pname).weight.data.copy_(merged.to(tensors[0].dtype))
        else:
            merged = sum(w.item() * t for w, t in zip(weights, tensors))
            getattr(centroid, pname).weight.data.copy_(merged)

        if getattr(centroid, pname).bias is not None:
            biases = [
                getattr(experts_module[i], pname).bias.data
                for i in all_indices
            ]
            if merge_float32:
                merged_b = sum(w.item() * b.float() for w, b in zip(weights, biases))
                getattr(centroid, pname).bias.data.copy_(merged_b.to(biases[0].dtype))
            else:
                merged_b = sum(w.item() * b for w, b in zip(weights, biases))
                getattr(centroid, pname).bias.data.copy_(merged_b)


# ---------------------------------------------------------------------------
# Observation helpers (single-layer and multi-layer)
# ---------------------------------------------------------------------------


def _count_layer_experts(layer: nn.Module, attrs: ModelAttrs) -> int:
    """Count how many experts a layer currently has."""
    experts = _resolve_path(layer, attrs.experts)
    if attrs.expert_storage == "tensor3d":
        return getattr(experts, attrs.gate_proj).shape[0]
    return len(experts)


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


@dataclass
class _LayerObservation:
    """All collected data for a single MoE layer."""

    scores: list[ExpertScore]
    router_logits: torch.Tensor | None
    # Subsampled data for expert output similarity
    hidden_sample: torch.Tensor | None
    topk_weights_sample: torch.Tensor | None
    topk_indices_sample: torch.Tensor | None


def _observe_single_layer(
    model: nn.Module,
    dataloader,
    layer: nn.Module,
    attrs: ModelAttrs,
    num_experts: int,
    top_k: int,
    device: torch.device,
    max_similarity_tokens: int = 2048,
) -> _LayerObservation:
    """Run calibration data and collect stats + router logits for one layer."""
    router = _resolve_path(layer, attrs.router)
    stats = _init_layer_stats(num_experts, device)
    router_logits_list: list[torch.Tensor] = []
    # Subsample hidden states + routing for expert output similarity
    hidden_samples: list[torch.Tensor] = []
    topk_w_samples: list[torch.Tensor] = []
    topk_i_samples: list[torch.Tensor] = []
    similarity_tokens_collected = 0

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def pre_hook(module, args):
        h = args[0] if isinstance(args, tuple) else args
        if isinstance(h, torch.Tensor):
            captured["hidden"] = h

    def post_hook(module, args, output):
        captured["router_output"] = output

    hooks.append(router.register_forward_pre_hook(pre_hook))
    hooks.append(router.register_forward_hook(post_hook))

    total_tokens = 0
    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                model(input_ids=input_ids, attention_mask=attn_mask)
                total_tokens += input_ids.numel()

                hidden = _to_2d(captured["hidden"])
                router_output = captured["router_output"]

                top_k_w, top_k_i = _parse_router_output(router_output, top_k)
                top_k_w = _to_2d(top_k_w)
                top_k_i = _to_2d(top_k_i)

                full_logits = _extract_full_logits(router_output)
                if full_logits is not None:
                    router_logits_list.append(full_logits)

                # Collect subsampled data for expert output similarity
                if similarity_tokens_collected < max_similarity_tokens:
                    hidden_samples.append(hidden.cpu())
                    topk_w_samples.append(top_k_w.cpu())
                    topk_i_samples.append(top_k_i.cpu())
                    similarity_tokens_collected += hidden.shape[0]

                experts_module = _resolve_path(layer, attrs.experts)
                _accumulate_layer(
                    stats, hidden, top_k_w, top_k_i,
                    experts_module, attrs, None,
                )

                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    scores = _stats_to_scores(stats, num_experts, total_tokens)
    router_logits = (
        torch.cat(router_logits_list, dim=0) if router_logits_list else None
    )
    hidden_sample = (
        torch.cat(hidden_samples, dim=0) if hidden_samples else None
    )
    topk_w_sample = (
        torch.cat(topk_w_samples, dim=0) if topk_w_samples else None
    )
    topk_i_sample = (
        torch.cat(topk_i_samples, dim=0) if topk_i_samples else None
    )
    return _LayerObservation(
        scores=scores,
        router_logits=router_logits,
        hidden_sample=hidden_sample,
        topk_weights_sample=topk_w_sample,
        topk_indices_sample=topk_i_sample,
    )


def _observe_all_layers(
    model: nn.Module,
    dataloader,
    layers: list[nn.Module],
    moe_layer_indices: list[int],
    attrs: ModelAttrs,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> list[_LayerObservation]:
    """Observe all MoE layers in a single forward pass (one-shot mode)."""
    all_stats = [_init_layer_stats(num_experts, device) for _ in moe_layer_indices]
    all_router_logits: list[list[torch.Tensor]] = [[] for _ in moe_layer_indices]

    captured: dict[int, dict] = {}
    hooks = []

    for stat_idx, layer_idx in enumerate(moe_layer_indices):
        router = _resolve_path(layers[layer_idx], attrs.router)
        si = stat_idx

        def _make_pre(si=si):
            def hook(module, args):
                h = args[0] if isinstance(args, tuple) else args
                if isinstance(h, torch.Tensor):
                    captured[si] = {"hidden": h}
            return hook

        def _make_post(si=si):
            def hook(module, args, output):
                captured.setdefault(si, {})["router_output"] = output
            return hook

        hooks.append(router.register_forward_pre_hook(_make_pre()))
        hooks.append(router.register_forward_hook(_make_post()))

    total_tokens = 0
    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                model(input_ids=input_ids, attention_mask=attn_mask)
                total_tokens += input_ids.numel()

                for stat_idx, layer_idx in enumerate(moe_layer_indices):
                    if stat_idx not in captured:
                        continue
                    cap = captured[stat_idx]
                    hidden = _to_2d(cap["hidden"])
                    router_output = cap["router_output"]

                    top_k_w, top_k_i = _parse_router_output(router_output, top_k)
                    top_k_w = _to_2d(top_k_w)
                    top_k_i = _to_2d(top_k_i)

                    full_logits = _extract_full_logits(router_output)
                    if full_logits is not None:
                        all_router_logits[stat_idx].append(full_logits)

                    experts_module = _resolve_path(
                        layers[layer_idx], attrs.experts
                    )
                    _accumulate_layer(
                        all_stats[stat_idx], hidden, top_k_w, top_k_i,
                        experts_module, attrs, None,
                    )

                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    results = []
    for stat_idx in range(len(moe_layer_indices)):
        scores = _stats_to_scores(all_stats[stat_idx], num_experts, total_tokens)
        rl = (
            torch.cat(all_router_logits[stat_idx], dim=0)
            if all_router_logits[stat_idx]
            else None
        )
        # One-shot mode doesn't collect hidden samples for expert output
        # similarity (would require too much memory across all layers)
        results.append(_LayerObservation(
            scores=scores,
            router_logits=rl,
            hidden_sample=None,
            topk_weights_sample=None,
            topk_indices_sample=None,
        ))

    return results


def _stats_to_scores(
    stats: LayerStats, num_experts: int, total_tokens: int
) -> list[ExpertScore]:
    """Convert accumulated LayerStats into ExpertScore list."""
    scores = []
    for expert_idx in range(num_experts):
        count = stats.count[expert_idx].item()
        if count > 0:
            score = stats.weighted_sum[expert_idx].item() / count
            freq = count / total_tokens
            avg_gate = stats.gate_sum[expert_idx].item() / count
            avg_norm = stats.activation_norm_sum[expert_idx].item() / count
        else:
            score = freq = avg_gate = avg_norm = 0.0

        scores.append(
            ExpertScore(
                layer_idx=0,
                expert_idx=expert_idx,
                score=score,
                frequency=freq,
                activation_norm=avg_norm,
                router_weight=avg_gate,
            )
        )
    return scores
