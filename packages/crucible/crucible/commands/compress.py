"""crucible compress — score experts, compress the model, save the result."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from crucible.commands.common import (
    load_expert_scores,
    load_model_and_tokenizer,
    resolve_arch_or_exit,
    resolve_device,
    resolve_dtype,
    serialize_expert_scores,
    serialize_shared_stats,
)
from crucible.methods.registry import (
    METHOD_REGISTRY,
    SCORER_REGISTRY,
    get_method,
    get_scorer,
    method_names,
    scorer_names,
)

NAME = "compress"
HELP = "Compress an MoE model"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--method",
        choices=method_names(),
        default="reap",
        help="Compression method (default: reap). "
             + "; ".join(f"{m.name} — {m.summary}" for m in METHOD_REGISTRY.values()),
    )
    parser.add_argument(
        "--scoring",
        choices=scorer_names(),
        default="reap",
        help="Expert scoring strategy (default: reap). "
             + "; ".join(f"{s.name} — {s.summary}" for s in SCORER_REGISTRY.values()),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.375,
        help="Fraction of experts to remove (default: 0.375 → 128→80)",
    )
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
        help="Calibration data profile (default/code-only/general). "
             "Ignored if --calibration is set.",
    )
    parser.add_argument(
        "--samples", type=int, default=512, help="Calibration samples (default: 512)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length (default: 2048)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Calibration batch size (default: 4)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--routing-aware",
        action="store_true",
        help="Adjust scores by routing disruption cost (protects hard-to-replace experts)",
    )
    parser.add_argument(
        "--routing-beta",
        type=float,
        default=1.0,
        help="Routing disruption weight (default: 1.0). Higher = stronger protection.",
    )
    parser.add_argument(
        "--scores-file",
        default=None,
        help="Path to scores.json from a prior compress/observe run. Skips "
             "scoring entirely (saves ~30 min). Use with --routing-aware to "
             "re-apply routing adjustment at the new target ratio.",
    )
    parser.add_argument(
        "--finetune-router",
        action="store_true",
        help="Fine-tune router projections after compression (~200 steps)",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=200,
        help="Router fine-tuning steps (default: 200)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, mps, cpu (default: auto)",
    )

    # Methods and scorers declare their own flags, so adding a strategy
    # doesn't mean editing this file.
    for method in METHOD_REGISTRY.values():
        if method.add_arguments is not None:
            method.add_arguments(parser)
    for scorer in SCORER_REGISTRY.values():
        if scorer.add_arguments is not None:
            scorer.add_arguments(parser)


def run(args) -> None:
    """Run the compression pipeline."""
    from crucible.data import build_calibration_dataloader
    from crucible.export import count_parameters, get_model_size_mb, save_compressed
    from crucible.methods.observer import _get_config_value
    from crucible.models.registry import get_model_attrs
    from crucible.types import MethodContext

    method = get_method(args.method)
    scorer = get_scorer(args.scoring)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)

    # --- Resolve architecture ---
    arch = resolve_arch_or_exit(model)
    attrs = get_model_attrs(arch)
    num_experts = _get_config_value(model.config, attrs.num_experts_key)
    top_k = _get_config_value(model.config, attrs.num_experts_per_tok_key)
    num_to_keep = int(num_experts * (1 - args.ratio))

    if num_to_keep < top_k:
        print(
            f"Error: ratio {args.ratio} leaves {num_to_keep} experts, "
            f"but model needs at least {top_k} (top_k)"
        )
        sys.exit(1)

    original_params = count_parameters(model)
    original_size = get_model_size_mb(model)

    print(f"\n{'='*60}")
    print(f"Model:       {args.model} ({arch})")
    print(f"Experts:     {num_experts} → {num_to_keep} ({args.ratio:.1%} reduction)")
    print(f"Method:      {method.name.upper()}")
    print(f"Parameters:  {original_params['total'] / 1e9:.2f}B")
    print(f"Size:        {original_size:.0f} MB ({dtype})")
    print(f"{'='*60}\n")

    # --- Build calibration data ---
    print("Loading calibration data...")
    t0 = time.time()
    # Resolve calibration data: explicit datasets > profile > default
    cal_datasets = args.calibration
    cal_profile = None
    if cal_datasets is None and args.calibration_profile != "default":
        cal_profile = args.calibration_profile
        print(f"  Using calibration profile: {cal_profile}")

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

    ctx = MethodContext(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        attrs=attrs,
        num_experts=num_experts,
        top_k=top_k,
        num_to_keep=num_to_keep,
        ratio=args.ratio,
        options={k: v for k, v in vars(args).items() if k not in ("command", "run")},
    )

    # --- Score experts ---
    observation = None
    if method.uses_scores:
        if args.scores_file:
            # Reusing scores skips the expensive observation pass. Routing-aware
            # (if set) is still applied fresh at the new target ratio.
            print(f"\nLoading scores from {args.scores_file}")
            ctx.scores = load_expert_scores(args.scores_file)
            print(f"  Loaded {len(ctx.scores)} layers "
                  f"x {len(ctx.scores[0])} experts")
        else:
            print(f"\nScoring experts (strategy: {scorer.name})...")
            t0 = time.time()
            scoring = scorer.score(ctx)
            ctx.scores = scoring.scores
            ctx.per_layer_keep = scoring.per_layer_keep
            observation = scoring.observation
            print(f"  Scoring done in {time.time() - t0:.1f}s")

        if args.routing_aware:
            from crucible.methods.routing_aware import adjust_for_routing_disruption

            print(f"\nAdjusting scores for routing disruption "
                  f"(beta={args.routing_beta})...")
            t0 = time.time()
            ctx.scores = adjust_for_routing_disruption(
                model, dataloader, attrs, ctx.scores,
                num_to_keep=num_to_keep, beta=args.routing_beta,
            )
            print(f"  Routing-aware adjustment done in {time.time() - t0:.1f}s")

    expert_scores_serial = (
        serialize_expert_scores(ctx.scores) if ctx.scores is not None else None
    )
    if expert_scores_serial is not None:
        _save_scratch_scores(args, expert_scores_serial, observation)

    # --- Compress ---
    print(f"\nRunning {method.name.upper()}...")
    t0 = time.time()
    method_info = method.compress(ctx).info

    compress_time = time.time() - t0
    compressed_params = count_parameters(model)
    compressed_size = get_model_size_mb(model)

    print(f"  Done in {compress_time:.1f}s")
    orig_b = original_params["total"] / 1e9
    comp_b = compressed_params["total"] / 1e9
    print(f"  Parameters: {orig_b:.2f}B → {comp_b:.2f}B")
    print(f"  Size: {original_size:.0f} MB → {compressed_size:.0f} MB")

    # --- Save compressed model FIRST (before optional fine-tuning) ---
    output_path = Path(args.output) / _output_name(args)

    print(f"\nSaving to {output_path}...")
    t0 = time.time()

    metadata = {
        "source_model": args.model,
        "method": args.method,
        "scoring": args.scoring,
        "compression_ratio": args.ratio,
        "original_experts": num_experts,
        "remaining_experts": num_to_keep,
        "calibration_profile": args.calibration_profile,
        "calibration_datasets": args.calibration or [args.calibration_profile],
        "calibration_samples": args.samples,
        "seed": args.seed,
        "original_params": original_params["total"],
        "compressed_params": compressed_params["total"],
        **method_info,
    }
    if expert_scores_serial is not None:
        metadata["expert_scores"] = expert_scores_serial

    save_compressed(
        model, tokenizer, output_path,
        compression_metadata=metadata,
        source_model_path=args.model,
    )
    print(f"  Saved in {time.time() - t0:.1f}s")

    # --- Router fine-tuning (optional, after save) ---
    if args.finetune_router:
        _finetune_router(args, model, tokenizer, dataloader, attrs,
                         output_path, metadata)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Output:     {output_path}")
    print(f"  Experts:    {num_experts} → {num_to_keep}")
    print(f"  Params:     {orig_b:.2f}B → {comp_b:.2f}B")
    print(f"  Size:       {original_size:.0f} MB → {compressed_size:.0f} MB")
    print("\nNext steps:")
    print("  # For local inference (Ollama, llama.cpp) — convert to GGUF:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_path}")
    quasi = "llama.cpp/build/bin/llama-quantize"
    print(f"  {quasi} {output_path}/model.gguf model-q4_k_m.gguf q4_k_m")
    print("  # For vLLM — quantize to compressed-tensors W4A16:")
    print(f"  crucible quantize {output_path} -o {output_path}-w4a16")
    print(f"{'='*60}")


def _output_name(args) -> str:
    model_name = args.model.split("/")[-1]
    scoring_tag = f"-{args.scoring}" if args.scoring != "reap" else ""
    return f"{model_name}-{args.method}{scoring_tag}-{int(args.ratio * 100)}pct"


def _save_scratch_scores(args, expert_scores_serial: list, observation) -> None:
    """Write the scores out before compressing.

    Observation is the expensive part of a run; a downstream bug (router shape
    mismatch, an OOM on save) must not throw it away.
    """
    scratch_dir = Path(args.output) / _output_name(args)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_scores = scratch_dir / "scores.json"

    payload = {
        "source_model": args.model,
        "scoring": args.scoring,
        "compression_ratio": args.ratio,
        "calibration_profile": args.calibration_profile,
        "calibration_samples": args.samples,
        "seed": args.seed,
        "expert_scores": expert_scores_serial,
    }
    if observation is not None and observation.shared_expert_stats:
        payload["shared_expert_stats"] = serialize_shared_stats(
            observation.shared_expert_stats
        )
        payload["total_tokens"] = observation.total_tokens

    with open(scratch_scores, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved pre-prune scores to {scratch_scores}")


def _finetune_router(args, model, tokenizer, dataloader, attrs,
                     output_path: Path, metadata: dict) -> None:
    from crucible.export import save_compressed
    from crucible.methods.finetune_router import finetune_router

    print(f"\nFine-tuning router ({args.finetune_steps} steps)...")
    t0 = time.time()

    try:
        ft_result = finetune_router(
            model, dataloader, attrs,
            steps=args.finetune_steps,
        )
        print(f"  Trainable params: {ft_result['trainable_params']:,}")
        print(f"  Loss: {ft_result['initial_loss']:.4f} → "
              f"{ft_result['final_loss']:.4f}")
        print(f"  Done in {time.time() - t0:.1f}s")

        # Save again with fine-tuned router
        ft_path = Path(str(output_path) + "-ft")
        print(f"  Saving fine-tuned model to {ft_path}...")
        metadata["router_finetuned"] = True
        metadata["finetune_steps"] = args.finetune_steps
        save_compressed(
            model, tokenizer, ft_path,
            compression_metadata=metadata,
            source_model_path=args.model,
        )
    except Exception as e:
        print(f"  WARNING: Router fine-tuning failed: {e}")
        print("  Compressed model (without fine-tuning) is already saved.")
