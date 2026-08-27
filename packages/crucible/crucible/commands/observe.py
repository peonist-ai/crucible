"""crucible observe — run REAP observation and save scores without compressing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crucible.commands.common import (
    load_model_and_tokenizer,
    resolve_arch_or_exit,
    resolve_device,
    resolve_dtype,
    serialize_expert_scores,
    serialize_shared_stats,
)

NAME = "observe"
HELP = "Run REAP observation and save scores without compressing"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--calibration",
        nargs="+",
        default=None,
        help="Calibration dataset(s). Omit to use --calibration-profile.",
    )
    parser.add_argument(
        "--calibration-profile",
        choices=["default", "code-only", "general"],
        default="default",
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-o", "--output",
        default="results/observation.json",
        help="Output JSON path (default: results/observation.json)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")


def run(args) -> None:
    from crucible.data import build_calibration_dataloader
    from crucible.methods.observer import (
        _get_config_value,
        compute_reap_scores,
        observe,
    )
    from crucible.models.registry import get_model_attrs

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)

    arch = resolve_arch_or_exit(model)
    attrs = get_model_attrs(arch)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    print(f"  arch={arch}  experts={num_experts}  top_k={top_k}")
    if attrs.shared_expert:
        print(f"  shared_expert={attrs.shared_expert} (will collect stats)")

    print("\nBuilding calibration dataloader...")
    t0 = time.time()
    cal_datasets = args.calibration
    cal_profile = args.calibration_profile if cal_datasets is None else None
    dataloader = build_calibration_dataloader(
        tokenizer,
        datasets=cal_datasets,
        profile=cal_profile,
        num_samples=args.samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  {len(dataloader.dataset)} samples, {len(dataloader)} batches "
          f"({time.time() - t0:.1f}s)")

    print("\nRunning observation...")
    t0 = time.time()
    result = observe(model, dataloader, attrs)
    print(f"  done in {time.time() - t0:.1f}s, {result.total_tokens} tokens")

    print("Computing REAP scores...")
    scores = compute_reap_scores(result)

    shared_serial = serialize_shared_stats(result.shared_expert_stats)
    layer_types = _classify_layers(model)

    payload = {
        "model": args.model,
        "architecture": arch,
        "num_experts": num_experts,
        "top_k": top_k,
        "total_tokens": result.total_tokens,
        "calibration_profile": args.calibration_profile,
        "calibration_datasets": args.calibration or [args.calibration_profile],
        "calibration_samples": args.samples,
        "seed": args.seed,
        "moe_layer_indices": result.moe_layer_indices,
        "layer_types": layer_types,
        "expert_scores": serialize_expert_scores(scores),
        "shared_expert_stats": shared_serial,
    }

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out}")
    _print_summary(scores, shared_serial, layer_types, result.moe_layer_indices)


def _classify_layers(model) -> list[str]:
    """Label each decoder layer 'linear' or 'full' by its attention class.

    Qwen 3.5/3.6 alternate 3x linear attention with 1x full attention, and
    prunability differs between them — worth recording alongside the scores.
    """
    from crucible.methods.observer import _find_layers

    layer_types: list[str] = []
    try:
        for layer in _find_layers(model):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                layer_types.append("unknown")
            else:
                name = type(attn).__name__.lower()
                layer_types.append(
                    "linear" if ("deltanet" in name or "linear" in name) else "full"
                )
    except Exception:
        return []
    return layer_types


def _print_summary(scores, shared_stats, layer_types, moe_indices) -> None:
    import statistics

    print("\n" + "=" * 70)
    print("OBSERVATION SUMMARY")
    print("=" * 70)

    flat = [s.score for ls in scores for s in ls if s.score > 0]
    if flat:
        print(f"\nGlobal nonzero scores (n={len(flat)}):")
        print(f"  min={min(flat):.4f}  max={max(flat):.4f}  mean={statistics.mean(flat):.4f}")
        q = statistics.quantiles(flat, n=10)
        print(f"  p10={q[0]:.4f}  p50={q[4]:.4f}  p90={q[8]:.4f}  "
              f"(p90/p10 = {q[8] / max(q[0], 1e-9):.2f}x)")

    # Per-layer CV, grouped by attention type
    cvs_by_type: dict[str, list[float]] = {}
    for li, layer_scores in enumerate(scores):
        layer_idx = moe_indices[li]
        ltype = layer_types[layer_idx] if layer_idx < len(layer_types) else "unknown"
        ss = [s.score for s in layer_scores if s.score > 0]
        if len(ss) >= 2:
            m = statistics.mean(ss)
            if m > 0:
                cvs_by_type.setdefault(ltype, []).append(statistics.stdev(ss) / m)
    if cvs_by_type:
        print("\nPer-layer coefficient of variation (higher = more prunable):")
        for t, cvs in cvs_by_type.items():
            print(f"  {t}: n={len(cvs)}  mean={statistics.mean(cvs):.3f}  "
                  f"range [{min(cvs):.3f}, {max(cvs):.3f}]")

    if shared_stats:
        print("\nShared expert contribution:")
        ratios = [s["shared_norm_over_mlp_norm"] for s in shared_stats.values()]
        gates = [s["gate_mean"] for s in shared_stats.values() if s["gate_mean"]]
        print(f"  shared_norm / mlp_norm: mean={statistics.mean(ratios):.3f}  "
              f"range [{min(ratios):.3f}, {max(ratios):.3f}]")
        if gates:
            print(f"  shared gate sigmoid: mean={statistics.mean(gates):.3f}  "
                  f"range [{min(gates):.3f}, {max(gates):.3f}]")
    print("=" * 70)
