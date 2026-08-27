"""Activation collection hooks for MoE expert analysis.

Collects router gate values and expert activation norms during a forward
pass over calibration data. Both REAP and REAM depend on this.

The observer hooks into the model's forward pass to capture:
  - Router outputs (gate values, top-k selection)
  - Hidden states at each MoE layer
Then manually computes per-expert activation norms for REAP scoring.

REAP saliency: S_j = (1/|X_j|) * sum_{x in X_j} g_j(x) * ||f_j(x)||_2
  where X_j = tokens selecting expert j, g = gate value, f = expert output
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from crucible.types import ExpertScore, ModelAttrs


@dataclass
class LayerStats:
    """Accumulated statistics for one MoE layer."""

    gate_sum: torch.Tensor  # [num_experts]
    activation_norm_sum: torch.Tensor  # [num_experts]
    weighted_sum: torch.Tensor  # [num_experts] gate * norm
    count: torch.Tensor  # [num_experts] selection count
    # Subsampled router logits/probs for REAM similarity computation
    router_logits: list[torch.Tensor] = field(default_factory=list)


@dataclass
class SharedExpertStats:
    """Per-layer shared-expert contribution stats.

    Shared experts run on every token regardless of routing — this tracks
    how load-bearing they are vs the routed experts. High shared contribution
    = more headroom to prune routed experts aggressively.
    """

    shared_norm_sum: torch.Tensor  # sum over tokens of ||shared_out||
    shared_norm_sq_sum: torch.Tensor
    gate_sum: torch.Tensor  # sum of sigmoid(gate_logit) per token
    gate_sq_sum: torch.Tensor
    mlp_out_norm_sum: torch.Tensor  # sum of ||full mlp out|| (shared + routed)
    token_count: torch.Tensor


@dataclass
class ObservationResult:
    """Complete observation results across all layers."""

    layer_stats: list[LayerStats]
    total_tokens: int
    num_experts: int
    top_k: int
    moe_layer_indices: list[int]
    shared_expert_stats: dict[int, SharedExpertStats] = field(default_factory=dict)


def observe(
    model: nn.Module,
    dataloader,
    attrs: ModelAttrs,
    *,
    store_router_logits: bool = False,
    max_router_logit_tokens: int = 10_000,
) -> ObservationResult:
    """Run calibration data through model and collect expert activation statistics.

    Hooks into the model's forward pass to capture router decisions and
    hidden states, then computes per-expert output norms for REAP scoring.

    Args:
        model: HuggingFace model (already on device, eval mode).
        dataloader: yields dicts with 'input_ids' and optionally 'attention_mask'.
        attrs: model architecture mapping from registry.
        store_router_logits: store subsampled router logits for REAM.
        max_router_logit_tokens: cap on stored router logit tokens per layer.

    Returns:
        ObservationResult with per-layer statistics.
    """
    layers = _find_layers(model)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    device = next(model.parameters()).device

    # Find which layers have MoE blocks
    moe_layer_indices = []
    for i, layer in enumerate(layers):
        try:
            _resolve_path(layer, attrs.router)
            _resolve_path(layer, attrs.experts)
            moe_layer_indices.append(i)
        except AttributeError:
            continue

    if not moe_layer_indices:
        raise ValueError(
            f"No MoE layers found with router='{attrs.router}', "
            f"experts='{attrs.experts}'"
        )

    stats = [_init_layer_stats(num_experts, device) for _ in moe_layer_indices]

    # Resolve the expert activation once, from the first MoE layer, so a missing
    # `hidden_act` warns a single time instead of once per expert per batch.
    first_experts = _resolve_path(layers[moe_layer_indices[0]], attrs.experts)
    act_probe = (
        first_experts
        if attrs.expert_storage == "tensor3d"
        else first_experts[0]
    )
    expert_act = _resolve_expert_activation(act_probe, attrs, model.config)

    # Register hooks on each MoE layer's router
    hooks = []
    captured: dict[int, dict] = {}

    for stat_idx, layer_idx in enumerate(moe_layer_indices):
        router = _resolve_path(layers[layer_idx], attrs.router)

        # Capture hidden states entering the router
        si = stat_idx  # bind for closure

        def _make_pre_hook(si=si):
            def hook(module, args):
                h = args[0] if isinstance(args, tuple) else args
                if isinstance(h, torch.Tensor):
                    captured[si] = {"hidden_states": h}

            return hook

        def _make_post_hook(si=si):
            def hook(module, args, output):
                captured.setdefault(si, {})["router_output"] = output

            return hook

        hooks.append(router.register_forward_pre_hook(_make_pre_hook()))
        hooks.append(router.register_forward_hook(_make_post_hook()))

    # Attach shared-expert hooks if the model has one (e.g., Qwen 3.6)
    shared_stats: dict[int, SharedExpertStats] = {}
    shared_capture: dict[int, dict] = {}
    if attrs.shared_expert is not None:
        hooks.extend(
            _attach_shared_expert_hooks(
                layers, attrs, shared_stats, shared_capture, device
            )
        )

    total_tokens = 0
    num_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
    model.eval()

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches and batch_idx % max(1, num_batches // 10) == 0:
                    pct = batch_idx * 100 // num_batches if num_batches else 0
                    print(f"      batch {batch_idx}/{num_batches} ({pct}%)", end="\r", flush=True)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                model(input_ids=input_ids, attention_mask=attention_mask)
                total_tokens += input_ids.numel()

                for stat_idx, layer_idx in enumerate(moe_layer_indices):
                    if stat_idx not in captured:
                        continue

                    cap = captured[stat_idx]
                    hidden = _to_2d(cap["hidden_states"])
                    router_output = cap["router_output"]

                    top_k_weights, top_k_index = _parse_router_output(
                        router_output, top_k
                    )
                    top_k_weights = _to_2d(top_k_weights)
                    top_k_index = _to_2d(top_k_index)

                    # Store subsampled router logits for REAM
                    router_logits_for_ream = None
                    if store_router_logits:
                        stored = sum(
                            t.shape[0] for t in stats[stat_idx].router_logits
                        )
                        if stored < max_router_logit_tokens:
                            router_logits_for_ream = _extract_full_logits(
                                router_output
                            )

                    experts_module = _resolve_path(
                        layers[layer_idx], attrs.experts
                    )
                    _accumulate_layer(
                        stats[stat_idx],
                        hidden,
                        top_k_weights,
                        top_k_index,
                        experts_module,
                        attrs,
                        router_logits_for_ream,
                        expert_act,
                    )

                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    return ObservationResult(
        layer_stats=stats,
        total_tokens=total_tokens,
        num_experts=num_experts,
        top_k=top_k,
        moe_layer_indices=moe_layer_indices,
        shared_expert_stats=shared_stats,
    )


def compute_reap_scores(result: ObservationResult) -> list[list[ExpertScore]]:
    """Compute REAP saliency scores from observation results.

    S_j = (1/|X_j|) * sum_{x in X_j} g_j(x) * ||f_j(x)||_2

    Averages over tokens where expert j was *selected* (not globally),
    decoupling impact from frequency. Rare specialists keep high scores.

    Returns:
        List of lists — scores[layer_idx][expert_idx].
    """
    all_scores = []
    for stat_idx, layer_stats in enumerate(result.layer_stats):
        layer_idx = result.moe_layer_indices[stat_idx]
        scores = []
        for expert_idx in range(result.num_experts):
            count = layer_stats.count[expert_idx].item()
            if count > 0:
                score = layer_stats.weighted_sum[expert_idx].item() / count
                freq = count / result.total_tokens
                avg_gate = layer_stats.gate_sum[expert_idx].item() / count
                avg_norm = (
                    layer_stats.activation_norm_sum[expert_idx].item() / count
                )
            else:
                score = 0.0
                freq = 0.0
                avg_gate = 0.0
                avg_norm = 0.0

            scores.append(
                ExpertScore(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    score=score,
                    frequency=freq,
                    activation_norm=avg_norm,
                    router_weight=avg_gate,
                )
            )
        all_scores.append(scores)
    return all_scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(module: nn.Module, dot_path: str) -> nn.Module:
    """Resolve a dot-separated attribute path on a module."""
    for part in dot_path.split("."):
        module = getattr(module, part)
    return module


def _find_layers(model: nn.Module) -> list[nn.Module]:
    """Find transformer decoder layers in a HuggingFace model."""
    for path in [
        "model.layers",
        "model.model.layers",
        "model.text_model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
    ]:
        try:
            return list(_resolve_path(model, path))
        except AttributeError:
            continue
    raise ValueError(
        "Cannot find transformer layers. Tried: model.layers, "
        "model.model.layers, model.text_model.layers, "
        "language_model.model.layers"
    )


def _get_config_value(config, key: str):
    """Get a config value, checking nested text_config for multimodal models."""
    sentinel = object()
    value = _get_config_value_or_none(config, key, default=sentinel)
    if value is sentinel:
        raise ValueError(f"Cannot find '{key}' in model config")
    return value


def _get_config_value_or_none(config, key: str, default=None):
    """Same lookup as `_get_config_value`, but for keys that may not exist.

    Optional architecture features (grouped routing, activation names) are
    absent from most configs, and absence is an answer rather than an error.
    """
    if hasattr(config, key):
        return getattr(config, key)
    if hasattr(config, "text_config") and hasattr(config.text_config, key):
        return getattr(config.text_config, key)
    return default


def _init_layer_stats(num_experts: int, device: torch.device) -> LayerStats:
    return LayerStats(
        gate_sum=torch.zeros(num_experts, device=device, dtype=torch.float64),
        activation_norm_sum=torch.zeros(
            num_experts, device=device, dtype=torch.float64
        ),
        weighted_sum=torch.zeros(
            num_experts, device=device, dtype=torch.float64
        ),
        count=torch.zeros(num_experts, device=device, dtype=torch.int64),
    )


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape to [tokens, dim] if 3D."""
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    return t


def _parse_router_output(
    output, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k gate weights and indices from router output.

    Handles:
      - Tuple of 3: (probs, top_k_weights, top_k_index) — Gemma4
      - Single tensor: raw logits — compute top-k + softmax
    """
    if isinstance(output, tuple) and len(output) >= 3:
        # Gemma4-style: router already computed top-k with renorm + per-expert scale
        return output[1], output[2]

    logits = output if isinstance(output, torch.Tensor) else output[0]
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    top_k_logits, top_k_index = torch.topk(logits, top_k, dim=-1)
    top_k_weights = F.softmax(top_k_logits, dim=-1)
    return top_k_weights, top_k_index


def _extract_full_logits(router_output) -> torch.Tensor | None:
    """Get full router logits/probs for REAM similarity, moved to CPU."""
    if isinstance(router_output, tuple) and len(router_output) >= 3:
        return _to_2d(router_output[0]).cpu()
    if isinstance(router_output, torch.Tensor):
        return _to_2d(router_output).cpu()
    return None


def _act_by_name(name: str):
    """Resolve an activation by its torch.nn.functional name."""
    fn = getattr(F, name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"Unknown expert activation '{name}'. Expected a name in "
            f"torch.nn.functional, e.g. 'silu' or 'gelu'."
        )
    return fn


def _resolve_expert_activation(
    experts_module: nn.Module,
    attrs: ModelAttrs,
    config=None,
) -> object:
    """Find the activation the expert FFN actually uses.

    Only the `tensor3d` path needs this: there is no per-expert module to call,
    so the FFN has to be recomputed by hand and the activation is the one part
    that cannot be read off the weight shapes. Guessing it is exactly the class
    of assumption that made REAM wrong on Gemma 4, so prefer what the model
    tells us, in descending order of authority:

      1. an explicit `ModelAttrs.expert_act` override
      2. the experts module's own `act_fn` (HF fused-expert modules carry one)
      3. the config's `hidden_act` / `hidden_activation`
      4. SiLU, with a warning — every model in the registry today is SiLU-gated,
         but a silent default here would mis-score a GELU model rather than fail.
    """
    if attrs.expert_act:
        return _act_by_name(attrs.expert_act)

    act = getattr(experts_module, "act_fn", None)
    if callable(act):
        return act

    if config is not None:
        for key in ("hidden_act", "hidden_activation"):
            name = _get_config_value_or_none(config, key)
            if isinstance(name, str):
                return _act_by_name(name)

    warnings.warn(
        f"{type(experts_module).__name__} exposes no act_fn and the config has "
        "no hidden_act; assuming SiLU-gated experts for activation-norm "
        "scoring. Set ModelAttrs.expert_act if that is wrong — the scores "
        "depend on it.",
        RuntimeWarning,
        stacklevel=2,
    )
    return F.silu


def _compute_expert_output(
    hidden: torch.Tensor,
    experts_module: nn.Module,
    attrs: ModelAttrs,
    expert_idx: int,
    act=None,
) -> torch.Tensor:
    """Compute one expert's FFN output.

    For `modulelist` storage the expert is a real module, so call its own
    forward — that is the model's code, not our reconstruction of it, so it
    stays correct for any activation, bias, or extra projection the family
    happens to use. Only fall back to hand-rolled projections when the module
    defines no forward of its own (mocks, and the odd weight-holder class).

    For `tensor3d` storage there is no per-expert module to call, so the gated
    FFN is recomputed from the stacked weights with the activation resolved by
    `_resolve_expert_activation`.
    """
    if attrs.expert_storage == "tensor3d":
        if act is None:
            act = _resolve_expert_activation(experts_module, attrs)
        down_proj = getattr(experts_module, attrs.down_proj)
        if attrs.fused_gate_up:
            gate_up_proj = getattr(experts_module, attrs.gate_proj)
            gate_up = F.linear(hidden, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
        else:
            gate = F.linear(hidden, getattr(experts_module, attrs.gate_proj)[expert_idx])
            up = F.linear(hidden, getattr(experts_module, attrs.up_proj)[expert_idx])
        return F.linear(act(gate) * up, down_proj[expert_idx])

    expert = experts_module[expert_idx]
    if _has_own_forward(expert):
        out = expert(hidden)
        return out[0] if isinstance(out, tuple) else out

    if act is None:
        act = _resolve_expert_activation(expert, attrs)
    gate = act(getattr(expert, attrs.gate_proj)(hidden))
    up = getattr(expert, attrs.up_proj)(hidden)
    return getattr(expert, attrs.down_proj)(gate * up)


def _has_own_forward(module: nn.Module) -> bool:
    """True if `module` defines its own forward rather than inheriting the stub."""
    return type(module).forward is not nn.Module.forward


def _attach_shared_expert_hooks(
    layers: list[nn.Module],
    attrs: ModelAttrs,
    stats: dict[int, "SharedExpertStats"],
    capture: dict[int, dict],
    device: torch.device,
) -> list:
    """Hook shared expert, shared-expert gate, and full mlp output per layer.

    Accumulates per-token output norms so we can measure how load-bearing
    the shared expert is vs routed experts (informs pruning headroom).
    """
    hooks = []
    for i, layer in enumerate(layers):
        try:
            shared = _resolve_path(layer, attrs.shared_expert)
        except AttributeError:
            continue

        stats[i] = SharedExpertStats(
            shared_norm_sum=torch.zeros(1, device=device, dtype=torch.float64),
            shared_norm_sq_sum=torch.zeros(1, device=device, dtype=torch.float64),
            gate_sum=torch.zeros(1, device=device, dtype=torch.float64),
            gate_sq_sum=torch.zeros(1, device=device, dtype=torch.float64),
            mlp_out_norm_sum=torch.zeros(1, device=device, dtype=torch.float64),
            token_count=torch.zeros(1, device=device, dtype=torch.int64),
        )

        gate_mod = None
        if attrs.shared_expert_gate:
            try:
                gate_mod = _resolve_path(layer, attrs.shared_expert_gate)
            except AttributeError:
                pass

        # The parent block (layer.mlp) is the first segment of shared_expert path
        mlp_path = attrs.shared_expert.rsplit(".", 1)[0] if "." in attrs.shared_expert else None

        def _shared_hook(_m, _inp, out, i=i):
            t = out if isinstance(out, torch.Tensor) else out[0]
            if t.dim() == 3:
                t = t.reshape(-1, t.shape[-1])
            capture.setdefault(i, {})["shared_out"] = t

        def _gate_hook(_m, _inp, out, i=i):
            t = out if isinstance(out, torch.Tensor) else out[0]
            if t.dim() == 3:
                t = t.reshape(-1, t.shape[-1])
            capture.setdefault(i, {})["gate_raw"] = t

        def _mlp_hook(_m, _inp, out, i=i):
            t = out if isinstance(out, torch.Tensor) else out[0]
            if t.dim() == 3:
                t = t.reshape(-1, t.shape[-1])
            capture.setdefault(i, {})["mlp_out"] = t
            _accumulate_shared(capture[i], stats[i])
            capture.pop(i, None)

        hooks.append(shared.register_forward_hook(_shared_hook))
        if gate_mod is not None:
            hooks.append(gate_mod.register_forward_hook(_gate_hook))
        if mlp_path:
            mlp = _resolve_path(layer, mlp_path)
            hooks.append(mlp.register_forward_hook(_mlp_hook))
    return hooks


def _accumulate_shared(cap: dict, st: "SharedExpertStats") -> None:
    """Fold one batch's shared-expert measurements into running stats."""
    if "shared_out" not in cap or "mlp_out" not in cap:
        return
    shared_norm = cap["shared_out"].float().norm(dim=-1).double()
    mlp_norm = cap["mlp_out"].float().norm(dim=-1).double()

    st.shared_norm_sum += shared_norm.sum()
    st.shared_norm_sq_sum += (shared_norm ** 2).sum()
    st.mlp_out_norm_sum += mlp_norm.sum()
    st.token_count += shared_norm.shape[0]

    if "gate_raw" in cap:
        gate = torch.sigmoid(cap["gate_raw"].float()).squeeze(-1).double()
        st.gate_sum += gate.sum()
        st.gate_sq_sum += (gate ** 2).sum()


def _accumulate_layer(
    stats: LayerStats,
    hidden_states: torch.Tensor,
    top_k_weights: torch.Tensor,
    top_k_index: torch.Tensor,
    experts_module: nn.Module,
    attrs: ModelAttrs,
    router_logits: torch.Tensor | None,
    expert_act=None,
) -> None:
    """Accumulate REAP statistics for one layer from one batch."""
    if router_logits is not None:
        stats.router_logits.append(router_logits)

    active_experts = top_k_index.unique()

    for expert_idx_t in active_experts:
        eidx = expert_idx_t.item()

        # Which tokens selected this expert, and with what gate value
        mask = top_k_index == eidx  # [tokens, top_k]
        token_mask = mask.any(dim=-1)  # [tokens]
        # Gate value for this expert per token (sum across top-k slots; at most one hit)
        expert_gates = (top_k_weights * mask.float()).sum(dim=-1)[
            token_mask
        ]  # [n]

        # Compute expert FFN output for selected tokens
        expert_output = _compute_expert_output(
            hidden_states[token_mask], experts_module, attrs, eidx, expert_act
        )
        expert_norms = expert_output.float().norm(dim=-1)  # [n]

        # Accumulate in float64 for numerical stability
        g = expert_gates.double()
        n = expert_norms.double()
        stats.gate_sum[eidx] += g.sum()
        stats.activation_norm_sum[eidx] += n.sum()
        stats.weighted_sum[eidx] += (g * n).sum()
        stats.count[eidx] += token_mask.sum()
