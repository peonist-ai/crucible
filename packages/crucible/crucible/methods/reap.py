"""REAP: Router-weighted Expert Activation Pruning.

Scores experts by gate-value * activation-norm, then removes the
lowest-scoring experts per layer. One-shot, no fine-tuning.

Reference: arxiv.org/abs/2510.13999

Usage:
    result = observe(model, dataloader, attrs)
    scores = compute_reap_scores(result)
    prune_result = prune(model, scores, attrs, num_experts_to_keep=96)
    # model is now modified in-place with fewer experts
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

from crucible.methods.observer import (
    _find_layers,
    _get_config_value,
    _get_config_value_or_none,
    _resolve_path,
)
from crucible.types import ExpertScore, ModelAttrs

# Bias driven into padded router rows so they lose every top-k contest. Router
# logits are O(1..10) in practice, so this is far outside the reachable range
# while staying well inside fp16's max (65504) for a bf16/fp16 router.
_PAD_LOGIT_BIAS = -1e4


@dataclass
class GroupedRouting:
    """A group-limited router's constraint, read from the model config.

    DeepSeek-V3 shaped: the router scores `n_group` groups of experts, keeps the
    `top_k_group` best groups, then picks top_k experts from within them. Pruning
    has to respect it — dropping experts globally can empty a group or leave the
    selected groups holding fewer than top_k experts, either of which makes the
    router's own forward invalid.
    """

    n_group: int
    top_k_group: int
    group_size: int


@dataclass
class PruneResult:
    """Result of a REAP pruning operation."""

    experts_kept: list[list[int]]
    experts_removed: list[list[int]]
    original_num_experts: int
    remaining_num_experts: int
    moe_layer_indices: list[int]
    # "masked" or "cloned" when a non-uniform budget needed padding, else None.
    pad_strategy: str | None = None
    # The grouped-routing constraint honoured during selection, if the model has one.
    grouped_routing: GroupedRouting | None = None


def prune(
    model: nn.Module,
    scores: list[list[ExpertScore]],
    attrs: ModelAttrs,
    num_experts_to_keep: int | list[int],
) -> PruneResult:
    """Prune experts from the model in-place using REAP scores.

    For each MoE layer, keeps the top-scoring experts and removes the rest.
    Updates expert weights, router projection, and model config.

    Ranking is global within a layer unless the family uses group-limited
    routing, in which case the budget is split evenly across expert groups so the
    router's group selection stays satisfiable — see `_select_experts`.

    A non-uniform budget leaves layers of different widths, which serving
    runtimes cannot load, so short layers are padded back up. Padding is not
    free and not always inert; `_pad_layer` documents the two strategies and
    `PruneResult.pad_strategy` reports which one ran.

    Args:
        model: HuggingFace model (modified in-place).
        scores: per-layer expert scores from compute_reap_scores.
        attrs: model architecture mapping.
        num_experts_to_keep: experts to retain per layer. Either a single int
            (uniform) or a list of ints (one per MoE layer, for non-uniform
            pruning from pathfinder).

    Returns:
        PruneResult with details of what was pruned.
    """
    layers = _find_layers(model)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    grouping = _resolve_grouped_routing(model.config, attrs, num_experts)

    # Normalize to per-layer list
    if isinstance(num_experts_to_keep, int):
        per_layer_keep = [num_experts_to_keep] * len(scores)
    else:
        per_layer_keep = num_experts_to_keep

    if len(per_layer_keep) != len(scores):
        raise ValueError(
            f"per_layer_keep length ({len(per_layer_keep)}) != "
            f"number of scored layers ({len(scores)})"
        )

    for i, k in enumerate(per_layer_keep):
        if k >= num_experts:
            raise ValueError(
                f"Layer {i}: num_experts_to_keep ({k}) must be < "
                f"num_experts ({num_experts})"
            )
        if k < top_k:
            raise ValueError(
                f"Layer {i}: num_experts_to_keep ({k}) must be >= "
                f"top_k ({top_k})"
            )
        if grouping is not None:
            _validate_grouped_keep(k, top_k, grouping, layer_label=f"Layer {i}")

    all_kept = []
    all_removed = []
    moe_layer_indices = []
    # Position (in post-prune indexing) of the least load-bearing kept expert per
    # layer — the clone source if padding has to fall back to duplication.
    clone_sources = []

    for li, layer_scores in enumerate(scores):
        layer_idx = layer_scores[0].layer_idx
        moe_layer_indices.append(layer_idx)

        keep, remove = _select_experts(
            layer_scores, per_layer_keep[li], grouping=grouping
        )
        all_kept.append(keep)
        all_removed.append(remove)
        clone_sources.append(_lowest_scoring_position(layer_scores, keep))

        layer = layers[layer_idx]
        _prune_experts(layer, keep, attrs)
        _prune_router(layer, keep, attrs)

    # For non-uniform pruning, pad smaller layers to match the largest so
    # serving runtimes (vLLM) that expect uniform expert counts across layers
    # can load the model.
    pad_strategy = None
    if len(set(per_layer_keep)) > 1:
        max_kept = max(per_layer_keep)
        for li, layer_idx in enumerate(moe_layer_indices):
            if per_layer_keep[li] < max_kept:
                pad_count = max_kept - per_layer_keep[li]
                strategy = _pad_layer(
                    layers[layer_idx], pad_count, attrs, clone_sources[li]
                )
                pad_strategy = strategy
        if pad_strategy == "cloned":
            warnings.warn(
                "Non-uniform budget padded by cloning experts: this router "
                "scores experts from weights alone, with no additive bias to "
                "mask padded rows out of top-k. Padded experts are therefore "
                "reachable, and duplicate a kept expert rather than being "
                "inert. Routing on padded layers is perturbed slightly; a "
                "uniform ratio avoids padding altogether.",
                RuntimeWarning,
                stacklevel=2,
            )
        _update_config(model.config, attrs.num_experts_key, max_kept)
    else:
        _update_config(model.config, attrs.num_experts_key, per_layer_keep[0])

    effective_keep = (
        max(per_layer_keep) if isinstance(num_experts_to_keep, list) else num_experts_to_keep
    )
    return PruneResult(
        experts_kept=all_kept,
        experts_removed=all_removed,
        original_num_experts=num_experts,
        remaining_num_experts=effective_keep,
        moe_layer_indices=moe_layer_indices,
        pad_strategy=pad_strategy,
        grouped_routing=grouping,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_grouped_routing(
    config, attrs: ModelAttrs, num_experts: int
) -> GroupedRouting | None:
    """Read the group-limited routing constraint, if this family has one.

    Returns None when the registry entry declares no grouping keys or the config
    does not carry them — the common case, and the one where experts are simply
    ranked globally within a layer.
    """
    if not attrs.n_group_key or not attrs.top_k_group_key:
        return None

    n_group = _get_config_value_or_none(config, attrs.n_group_key)
    top_k_group = _get_config_value_or_none(config, attrs.top_k_group_key)
    if not n_group or not top_k_group or n_group <= 1:
        return None

    if num_experts % n_group != 0:
        raise ValueError(
            f"num_experts ({num_experts}) is not divisible by "
            f"{attrs.n_group_key} ({n_group}); cannot infer expert groups"
        )
    if top_k_group > n_group:
        raise ValueError(
            f"{attrs.top_k_group_key} ({top_k_group}) exceeds "
            f"{attrs.n_group_key} ({n_group})"
        )
    return GroupedRouting(
        n_group=n_group,
        top_k_group=top_k_group,
        group_size=num_experts // n_group,
    )


def _validate_grouped_keep(
    num_to_keep: int, top_k: int, grouping: GroupedRouting, layer_label: str
) -> None:
    """Check a keep-count can be met without breaking grouped routing.

    Two things have to hold. Groups must stay equal-sized, so the budget has to
    divide by the group count. And the router draws all top_k experts from just
    `top_k_group` groups, so those groups must between them still hold top_k
    experts — otherwise the router's own forward cannot fill its selection.
    """
    n_group = grouping.n_group
    if num_to_keep % n_group != 0:
        nearest = max(n_group, round(num_to_keep / n_group) * n_group)
        raise ValueError(
            f"{layer_label}: num_experts_to_keep ({num_to_keep}) must be "
            f"divisible by the router's group count ({n_group}) so expert "
            f"groups stay equal-sized. Nearest valid count: {nearest}."
        )

    per_group = num_to_keep // n_group
    reachable = per_group * grouping.top_k_group
    if reachable < top_k:
        raise ValueError(
            f"{layer_label}: keeping {num_to_keep} experts leaves {per_group} "
            f"per group, so the {grouping.top_k_group} groups the router "
            f"selects hold only {reachable} experts — fewer than top_k "
            f"({top_k}). Keep at least "
            f"{-(-top_k // grouping.top_k_group) * n_group}."
        )


def _select_experts(
    layer_scores: list[ExpertScore],
    num_to_keep: int,
    grouping: GroupedRouting | None = None,
) -> tuple[list[int], list[int]]:
    """Select which experts to keep based on scores.

    Without grouping, this is a plain global ranking within the layer. With a
    group-limited router, the same number is kept from *each* group, so groups
    stay equal-sized and every group the router can select stays populated.
    Ranking is still by saliency — only the budget is partitioned.

    Returns (kept_indices, removed_indices), both sorted ascending.
    """
    if grouping is None:
        ranked = sorted(layer_scores, key=lambda s: s.score, reverse=True)
        keep = sorted(s.expert_idx for s in ranked[:num_to_keep])
        remove = sorted(s.expert_idx for s in ranked[num_to_keep:])
        return keep, remove

    per_group = num_to_keep // grouping.n_group
    by_index = {s.expert_idx: s for s in layer_scores}
    keep: list[int] = []
    remove: list[int] = []
    for g in range(grouping.n_group):
        lo = g * grouping.group_size
        members = [
            by_index[i] for i in range(lo, lo + grouping.group_size) if i in by_index
        ]
        ranked = sorted(members, key=lambda s: s.score, reverse=True)
        keep.extend(s.expert_idx for s in ranked[:per_group])
        remove.extend(s.expert_idx for s in ranked[per_group:])
    return sorted(keep), sorted(remove)


def _lowest_scoring_position(
    layer_scores: list[ExpertScore], keep_indices: list[int]
) -> int:
    """Position of the lowest-scoring kept expert, in post-prune indexing.

    Used as the clone source when padding cannot mask experts out of routing.
    The lowest-scoring survivor is the right twin to make: it is selected least
    often and with the smallest gate values, so a duplicate of it perturbs
    routing less than a duplicate of a load-bearing expert would.
    """
    by_index = {s.expert_idx: s.score for s in layer_scores}
    return min(
        range(len(keep_indices)),
        key=lambda pos: by_index.get(keep_indices[pos], 0.0),
    )


def _expert_param_names(attrs: ModelAttrs) -> list[str]:
    """The distinct stacked-parameter names on a `tensor3d` experts module.

    Deduplicates rather than branching on `fused_gate_up`: a fused model names
    the same tensor for gate and up, an unfused one names three. Iterating the
    deduplicated set means an unfused 3D model gets its `up_proj` sliced too,
    which a hardcoded (gate, down) pair would silently skip.
    """
    ordered = {}
    for name in (attrs.gate_proj, attrs.up_proj, attrs.down_proj):
        ordered[name] = None
    return list(ordered)


def _prune_experts(
    layer: nn.Module, keep_indices: list[int], attrs: ModelAttrs
) -> None:
    """Remove pruned expert weights from a layer."""
    keep = torch.tensor(keep_indices, dtype=torch.long)

    if attrs.expert_storage == "tensor3d":
        experts = _resolve_path(layer, attrs.experts)

        for name in _expert_param_names(attrs):
            proj = getattr(experts, name)
            setattr(experts, name, nn.Parameter(proj.data[keep]))

        if hasattr(experts, "num_experts"):
            experts.num_experts = len(keep_indices)
    else:
        # ModuleList: replace with new list of kept experts
        parent, child_name = _resolve_parent(layer, attrs.experts)
        old_experts = getattr(parent, child_name)
        new_experts = nn.ModuleList([old_experts[i] for i in keep_indices])
        setattr(parent, child_name, new_experts)


def _prune_router(
    layer: nn.Module, keep_indices: list[int], attrs: ModelAttrs
) -> None:
    """Remove pruned expert rows from the router projection.

    Handles three router shapes:
      1. nn.Linear router (most models)
      2. Router module containing an nn.Linear child (Gemma4 etc.)
      3. Router module with a raw nn.Parameter `weight` (Qwen3_5MoeTopKRouter)
    """
    router = _resolve_path(layer, attrs.router)
    keep = torch.tensor(keep_indices, dtype=torch.long)
    new_num = len(keep_indices)

    linear = _find_router_linear_or_none(router)
    if linear is not None:
        linear.weight = nn.Parameter(linear.weight.data[keep])
        linear.out_features = new_num
        if linear.bias is not None:
            linear.bias = nn.Parameter(linear.bias.data[keep])
    elif hasattr(router, "weight") and isinstance(router.weight, nn.Parameter):
        # Parameter-based router (Qwen3_5MoeTopKRouter): weight is [num_experts, hidden]
        router.weight = nn.Parameter(router.weight.data[keep])
    else:
        raise ValueError(
            f"Router has no Linear projection or weight Parameter ({type(router).__name__})"
        )

    if hasattr(router, "num_experts"):
        router.num_experts = new_num

    # Update per-expert scale if present (Gemma4)
    if hasattr(router, "per_expert_scale"):
        router.per_expert_scale = nn.Parameter(
            router.per_expert_scale.data[keep]
        )

    # Additive score-correction bias, if the family carries one (DeepSeek's
    # `e_score_correction_bias`). It is per-expert, so it has to be sliced with
    # the router rows or routing scores misalign with the surviving experts.
    _slice_router_score_bias(router, attrs, keep)


def _slice_router_score_bias(
    router: nn.Module, attrs: ModelAttrs, keep: torch.Tensor
) -> None:
    """Slice the router's per-expert score-correction bias, if declared."""
    if not attrs.router_score_bias:
        return
    holder = getattr(router, attrs.router_score_bias, None)
    if holder is None:
        return
    sliced = holder.data[keep]
    if isinstance(holder, nn.Parameter):
        setattr(router, attrs.router_score_bias, nn.Parameter(sliced))
    else:
        router.register_buffer(attrs.router_score_bias, sliced)


def _pad_layer(
    layer: nn.Module, pad_count: int, attrs: ModelAttrs, clone_pos: int
) -> str:
    """Pad one layer up to the run's uniform expert count.

    Serving runtimes read a single `num_experts` from the config, so a
    non-uniform per-layer budget has to be padded back to a rectangle. There are
    two ways to do that and which one is available depends on the router:

    **masked** — when the router has an additive per-expert bias (a Linear bias,
    or a declared score-correction bias), padded rows get zero weights and a
    large negative bias. Their logit is then pinned far below every real expert,
    so top-k provably never selects them and their zero weights never run.

    **cloned** — when it does not. A zero weight row is *not* unselectable: its
    logit is exactly 0, and real logits are frequently negative, so a zero-padded
    expert wins top-k routinely and contributes a zero vector — a silent hole in
    the layer. With no bias to push the logit down there is no way to mask a row
    out of a linear router, so the padded expert is made an exact twin of a kept
    one instead. If routing picks it, the token gets a real expert's output.

    Returns the strategy used.
    """
    strategy = _pad_router(layer, pad_count, attrs, clone_pos)
    _pad_experts(
        layer, pad_count, attrs, clone_pos if strategy == "cloned" else None
    )
    return strategy


def _pad_experts(
    layer: nn.Module,
    pad_count: int,
    attrs: ModelAttrs,
    clone_pos: int | None,
) -> None:
    """Append `pad_count` dummy experts to a layer.

    `clone_pos` names a kept expert (in post-prune indexing) to duplicate; None
    means pad with zeros, which is only safe when the router masks the padded
    rows out of selection. See `_pad_layer`.
    """
    if attrs.expert_storage == "tensor3d":
        experts = _resolve_path(layer, attrs.experts)
        for name in _expert_param_names(attrs):
            proj = getattr(experts, name)
            if clone_pos is None:
                pad = torch.zeros(
                    pad_count, *proj.shape[1:],
                    dtype=proj.dtype, device=proj.device,
                )
            else:
                pad = proj.data[clone_pos].unsqueeze(0).repeat(
                    pad_count, *([1] * (proj.dim() - 1))
                )
            setattr(
                experts, name, nn.Parameter(torch.cat([proj.data, pad], dim=0))
            )
        if hasattr(experts, "num_experts"):
            experts.num_experts += pad_count
        return

    # ModuleList: append copies of a kept expert, zeroed when masking.
    parent, child_name = _resolve_parent(layer, attrs.experts)
    experts = getattr(parent, child_name)
    source = experts[clone_pos if clone_pos is not None else 0]
    padded = list(experts)
    for _ in range(pad_count):
        dup = copy.deepcopy(source)
        if clone_pos is None:
            with torch.no_grad():
                for p in dup.parameters():
                    p.zero_()
        padded.append(dup)
    setattr(parent, child_name, nn.ModuleList(padded))


def _pad_router(
    layer: nn.Module,
    pad_count: int,
    attrs: ModelAttrs,
    clone_pos: int,
) -> str:
    """Extend the router by `pad_count` experts, masking them if possible.

    Handles the same three router shapes as `_prune_router`, including the
    Parameter-based routers (`Qwen3_5MoeTopKRouter`) that have no Linear child —
    padding used to hard-require a Linear and raise on those, which made every
    non-uniform budget unreachable on Qwen 3.5/3.6.

    Returns "masked" or "cloned", per `_pad_layer`.
    """
    router = _resolve_path(layer, attrs.router)
    linear = _find_router_linear_or_none(router)

    if linear is not None:
        weight = linear.weight
    elif hasattr(router, "weight") and isinstance(router.weight, nn.Parameter):
        weight = router.weight
    else:
        raise ValueError(
            f"Router has no Linear projection or weight Parameter "
            f"({type(router).__name__}); cannot pad it"
        )

    bias_owner, bias_attr = _router_additive_bias(router, linear, attrs)
    strategy = "masked" if bias_owner is not None else "cloned"

    if strategy == "masked":
        pad_weight = torch.zeros(
            pad_count, weight.shape[1], dtype=weight.dtype, device=weight.device
        )
    else:
        pad_weight = weight.data[clone_pos].unsqueeze(0).repeat(pad_count, 1)

    new_weight = nn.Parameter(torch.cat([weight.data, pad_weight], dim=0))
    if linear is not None:
        linear.weight = new_weight
        linear.out_features += pad_count
    else:
        router.weight = new_weight

    if bias_owner is not None:
        bias = getattr(bias_owner, bias_attr)
        pad_bias = torch.full(
            (pad_count,), _PAD_LOGIT_BIAS, dtype=bias.dtype, device=bias.device
        )
        padded = torch.cat([bias.data, pad_bias], dim=0)
        if isinstance(bias, nn.Parameter):
            setattr(bias_owner, bias_attr, nn.Parameter(padded))
        else:
            bias_owner.register_buffer(bias_attr, padded)

    if hasattr(router, "num_experts"):
        router.num_experts += pad_count

    # Per-expert scale (Gemma4). Under masking, zero it as a second line of
    # defence. Under cloning, copy the twin's source so it behaves identically —
    # the scale is multiplicative, not an additive logit, so it cannot be relied
    # on to keep a padded expert out of top-k.
    if hasattr(router, "per_expert_scale"):
        scale = router.per_expert_scale
        if strategy == "masked":
            pad_scale = torch.zeros(
                pad_count, dtype=scale.dtype, device=scale.device
            )
        else:
            pad_scale = scale.data[clone_pos].unsqueeze(0).repeat(pad_count)
        router.per_expert_scale = nn.Parameter(
            torch.cat([scale.data, pad_scale], dim=0)
        )

    return strategy


def _router_additive_bias(
    router: nn.Module, linear: nn.Linear | None, attrs: ModelAttrs
) -> tuple[nn.Module | None, str]:
    """Find an additive per-expert term we can drive negative to mask a row.

    Returns (owner_module, attribute_name), or (None, "") when the router scores
    experts with weights alone and padded rows therefore cannot be masked.
    """
    if attrs.router_score_bias:
        holder = getattr(router, attrs.router_score_bias, None)
        if holder is not None:
            return router, attrs.router_score_bias
    if linear is not None and linear.bias is not None:
        return linear, "bias"
    return None, ""


def _find_router_linear(router: nn.Module) -> nn.Linear:
    """Find the Linear projection in the router (raises if none).

    Kept for callers that genuinely need a Linear — router fine-tuning trains
    one. Pruning and padding use the `_or_none` form and handle the
    Parameter-based routers too.
    """
    linear = _find_router_linear_or_none(router)
    if linear is None:
        raise ValueError(
            f"Cannot find Linear projection in router ({type(router).__name__})"
        )
    return linear


def _find_router_linear_or_none(router: nn.Module) -> nn.Linear | None:
    """Find the Linear projection in the router, or None if router is
    Parameter-based (e.g. Qwen3_5MoeTopKRouter uses a raw nn.Parameter)."""
    if isinstance(router, nn.Linear):
        return router
    for _name, module in router.named_children():
        if isinstance(module, nn.Linear):
            return module
    return None


def _resolve_parent(
    module: nn.Module, dot_path: str
) -> tuple[nn.Module, str]:
    """Resolve a dot-path and return (parent_module, child_attr_name)."""
    if "." in dot_path:
        parent_path, child_name = dot_path.rsplit(".", 1)
        return _resolve_path(module, parent_path), child_name
    return module, dot_path


def _update_config(config, num_experts_key: str, new_num: int) -> None:
    """Update the model config with new expert count."""
    if hasattr(config, num_experts_key):
        setattr(config, num_experts_key, new_num)
    elif hasattr(config, "text_config") and hasattr(
        config.text_config, num_experts_key
    ):
        setattr(config.text_config, num_experts_key, new_num)
