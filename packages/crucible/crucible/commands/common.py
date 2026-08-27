"""Helpers shared by more than one subcommand.

Torch and transformers are imported inside the functions that need them: the
CLI builds its whole parser before it knows which command will run, and
paying for those imports on `crucible --help` is a second of nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from crucible.types import ExpertScore


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device_str != "auto":
        return device_str

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_str: str):
    """Resolve dtype string to torch dtype."""
    import torch

    if dtype_str == "auto":
        return torch.bfloat16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def resolve_arch(model) -> str | None:
    """Find the registry arch name: prefer config.architectures, then fall back
    to the runtime model class name. AutoModelForCausalLM on a multimodal
    model (e.g. Qwen3_5MoeForConditionalGeneration) can load as the causal-LM
    variant with architectures=None, so runtime class is the reliable source.
    """
    from crucible.models.registry import MODEL_REGISTRY

    for a in (model.config.architectures or []):
        if a in MODEL_REGISTRY:
            return a
    cls = type(model).__name__
    if cls in MODEL_REGISTRY:
        return cls
    return None


def resolve_arch_or_exit(model) -> str:
    """resolve_arch, but print the supported list and exit if there's no match."""
    from crucible.models.registry import MODEL_REGISTRY

    arch = resolve_arch(model)
    if arch is None:
        print(f"Unsupported architecture: {model.config.architectures} "
              f"(runtime class: {type(model).__name__})")
        print(f"Supported: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)
    return arch


def load_model_and_tokenizer(model_id: str, device: str, dtype):
    """Load a model for compression or observation, and its tokenizer.

    Loads to CPU first, then moves to the GPU as a single transfer. Loading
    directly with device_map="cuda" fragments the allocator on Strix Halo
    (693 weight shards x small allocations hits a segment limit long before
    we run out of real memory). CPU load + single .to() avoids this.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} on {device} ({dtype})...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        print(f"  moving to {device}...")
        t1 = time.time()
        model = model.to(device)
        print(f"  moved in {time.time() - t1:.1f}s")

    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def serialize_expert_scores(scores: list[list[ExpertScore]]) -> list[list[dict]]:
    """Flatten per-layer ExpertScores into JSON-friendly dicts."""
    return [
        [
            {
                "layer_idx": s.layer_idx,
                "expert_idx": s.expert_idx,
                "score": s.score,
                "frequency": s.frequency,
                "activation_norm": s.activation_norm,
                "router_weight": s.router_weight,
            }
            for s in layer_scores
        ]
        for layer_scores in scores
    ]


def load_expert_scores(path: str) -> list[list[ExpertScore]]:
    """Read scores saved by a prior compress/observe run."""
    with open(Path(path).expanduser()) as f:
        saved = json.load(f)
    return [
        [
            ExpertScore(
                layer_idx=e["layer_idx"],
                expert_idx=e["expert_idx"],
                score=e["score"],
                frequency=e.get("frequency", 0.0),
                activation_norm=e.get("activation_norm", 0.0),
                router_weight=e.get("router_weight", 0.0),
            )
            for e in layer_scores
        ]
        for layer_scores in saved["expert_scores"]
    ]


def serialize_shared_stats(shared_expert_stats: dict) -> dict[str, dict[str, Any]]:
    """Convert SharedExpertStats tensors to JSON-friendly per-layer dict.

    Keys: token_count, shared_norm_mean/std, gate_mean/std, mlp_out_norm_mean,
    shared_norm_over_mlp_norm (approx. how much the shared expert dominates),
    effective_shared_contribution (shared_norm scaled by sigmoid gate value).
    """
    out: dict[str, dict] = {}
    for layer_idx, st in shared_expert_stats.items():
        n = st.token_count.item()
        if n == 0:
            continue
        sn = st.shared_norm_sum.item() / n
        sn_var = max(0.0, st.shared_norm_sq_sum.item() / n - sn ** 2)
        g_total = st.gate_sum.item()
        g = g_total / n if g_total else 0.0
        g_var = max(0.0, st.gate_sq_sum.item() / n - g ** 2) if g else 0.0
        mlp = st.mlp_out_norm_sum.item() / n
        out[str(layer_idx)] = {
            "token_count": n,
            "shared_norm_mean": sn,
            "shared_norm_std": sn_var ** 0.5,
            "gate_mean": g,
            "gate_std": g_var ** 0.5,
            "mlp_out_norm_mean": mlp,
            "shared_norm_over_mlp_norm": sn / mlp if mlp > 0 else 0.0,
            "effective_shared_contribution": sn * g if g else sn,
        }
    return out
