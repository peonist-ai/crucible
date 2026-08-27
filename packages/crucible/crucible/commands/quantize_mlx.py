"""crucible quantize-mlx — third output contract: pruned weights to MLX.

`crucible quantize` targets vLLM on ROCm; this targets a Mac. The two share the
calibration mix and nothing else — MLX quantization is weight surgery, so none of
the offload/device-map/attention-implementation machinery applies, which is why
this is its own command rather than a `--format` flag on that one.

Requires mlx-lm, which crucible does not depend on. See `crucible.quantize_mlx`
for the install line for each backend.
"""

from __future__ import annotations

import argparse

NAME = "quantize-mlx"
HELP = "Quantize a model for MLX / Apple silicon (needs mlx-lm)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Path to HF model (original or REAP-compressed)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <model>-mlx-<bits>bit)")

    width = parser.add_argument_group("bit allocation")
    width.add_argument(
        "--bits", type=int, default=4,
        help="Width for expert/FFN weights — nearly all the model (default: 4). "
             "This is the knob that decides whether it fits: at 4 bits a "
             "19.2B-parameter REAP model lands near 11GB, at 3 bits near 8.5GB.",
    )
    width.add_argument(
        "--group-size", type=int, default=64,
        help="Scale/bias group size along the input dim (default: 64). MLX skips "
             "any module whose last dim is not divisible by this, silently "
             "leaving it in full precision — 64 divides every dim in the Qwen "
             "3.5/3.6 and Gemma 4 MoE blocks.",
    )
    width.add_argument(
        "--high-bits", type=int, default=8,
        help="Width for attention projections and the embedding/output tensors "
             "(default: 8). This is the split the shipped Q3K-mixed GGUF uses: "
             "in a 35B-A3B almost all the weight is expert weight, so holding "
             "attention high costs little size and buys disproportionate "
             "quality.",
    )
    width.add_argument(
        "--high-group-size", type=int, default=64,
        help="Group size for the high-precision modules (default: 64).",
    )
    width.add_argument(
        "--uniform", action="store_true",
        help="Drop the high/low split and quantize everything at --bits. The "
             "model's own protections (e.g. 8-bit routers on Qwen 3.5/3.6 MoE) "
             "still apply — those are never ours to override.",
    )
    width.add_argument(
        "--recipe", default=None, choices=list(_recipes()),
        help="Use one of mlx-lm's own mixed schemes instead of ours. These "
             "reproduce llama.cpp's Q4_K_M heuristic (more bits in the first and "
             "last eighth of the stack, and on down_proj/v_proj). Overrides "
             "--high-bits.",
    )
    width.add_argument(
        "--mode", default="affine",
        choices=("affine", "mxfp4", "nvfp4", "mxfp8"),
        help="Quantization mode (default: affine). The float modes ignore "
             "--bits/--group-size and use their own fixed layouts.",
    )

    method = parser.add_argument_group("method")
    method.add_argument(
        "--method", default="rtn", choices=("rtn", "gptq"),
        help="rtn (default): round-to-nearest, no forward pass — seconds to "
             "minutes, runs anywhere including a CPU-only MLX build. gptq: "
             "error-compensating, needs forward passes over calibration data "
             "and a backend that can actually run them. AWQ and DWQ are "
             "deliberately absent: mlx-lm's AWQ raises NotImplementedError for "
             "qwen3_5_moe (no entry in AWQ_MODEL_CONFIGS), and DWQ needs "
             "gradients through the full model.",
    )
    method.add_argument(
        "--fallback-bits", type=int, default=8,
        help="GPTQ only: width for modules GPTQ does not handle (default: 8).",
    )

    calib = parser.add_argument_group("calibration (--method gptq only)")
    calib.add_argument(
        "--samples", type=int, default=128,
        help="Number of calibration WINDOWS of --sequence-length tokens "
             "(default: 128).",
    )
    calib.add_argument(
        "--sequence-length", type=int, default=512,
        help="Window size for calibration (default: 512, matching mlx-lm). "
             "Windows are cut from whole rendered conversations and only full "
             "ones are kept, so this trades window count against context depth.",
    )
    calib.add_argument(
        "--calibration-profile", default="default",
        choices=["default", "code-only", "general"],
        help="Calibration mix (default: 'default' — the coding-agent mix shared "
             "with the expert scorer and the imatrix pass). Substituted for "
             "mlx-lm's own generic web-text calibration file.",
    )
    calib.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the bit allocation and the projected size, then exit "
             "without quantizing. Worth doing first when the real run is an "
             "hour of CPU.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Execute custom modeling code shipped with the checkpoint. This "
             "runs arbitrary Python from the model repo — only enable it for "
             "sources you trust.",
    )


def _recipes():
    from crucible.quantize_mlx import MLX_RECIPES

    return MLX_RECIPES


def run(args) -> None:
    import json
    import time
    from pathlib import Path

    from crucible.quantize_mlx import (
        MTP_NOTE,
        build_calibration_tokens,
        estimate_size_gb,
        gptq_router_modules,
        load_for_quantization,
        plan_quantization,
        require_mlx,
        resolve_predicate,
    )

    mlx = require_mlx()

    model_path = args.model
    suffix = f"-mlx-{args.mode}" if args.mode != "affine" else f"-mlx-{args.bits}bit"
    output_dir = Path(args.output or (model_path.rstrip("/") + suffix))

    if output_dir.exists() and not args.dry_run:
        raise SystemExit(
            f"{output_dir} already exists — remove it or pass a different -o. "
            "Refusing to write into a populated model directory."
        )

    print(f"Loading {model_path} (lazy)...")
    t0 = time.time()
    model, tokenizer, config = load_for_quantization(
        mlx, model_path, trust_remote_code=args.trust_remote_code
    )
    print(f"  loaded in {time.time() - t0:.1f}s  (model_type={config.get('model_type')})")

    predicate = resolve_predicate(
        model,
        recipe=args.recipe,
        bits=args.bits,
        group_size=args.group_size,
        high_bits=args.high_bits,
        high_group_size=args.high_group_size,
        keep_attention_high=not args.uniform,
        keep_embeddings_high=not args.uniform,
    )

    plan = plan_quantization(
        model, predicate, bits=args.bits, group_size=args.group_size
    )
    print("\nBit allocation:")
    for bits, entry in sorted(plan["by_bits"].items()):
        share = 100 * entry["params"] / max(1, plan["total_params"])
        print(f"  {bits:>2}-bit  {entry['modules']:>5} modules  "
              f"{entry['params'] / 1e9:>7.3f}B params  ({share:>5.1f}%)")
    if plan["unquantized_params"]:
        print(f"  16-bit  {plan['unquantized_modules']:>5} modules  "
              f"{plan['unquantized_params'] / 1e9:>7.3f}B params  (norms, biases, "
              "and anything whose last dim is not divisible by --group-size)")
    print(f"\n  projected: {plan['bits_per_weight']:.3f} bits/weight, "
          f"~{estimate_size_gb(plan['total_params'], plan['bits_per_weight']):.2f} GiB "
          f"over {plan['total_params'] / 1e9:.2f}B parameters")

    if args.method == "gptq":
        protected = gptq_router_modules(model)
        if protected:
            print(f"\n  WARNING: --method gptq ignores the model's own quant "
                  f"predicate and will quantize {len(protected)} protected "
                  f"module(s) at {args.bits} bits, including "
                  f"{protected[0].rsplit('.', 1)[-1]!r}. On a MoE those are the "
                  "routers: a noisy router changes WHICH expert runs, which is a "
                  "worse failure than a noisy expert. Use --method rtn unless you "
                  "have measured that this is worth it.")

    if args.dry_run:
        print("\n--dry-run: stopping before quantization.")
        return

    calibration = None
    t0 = time.time()
    if args.method == "gptq":
        print(f"\nBuilding calibration set ({args.calibration_profile} mix)...")
        calibration = build_calibration_tokens(
            tokenizer,
            profile=args.calibration_profile,
            num_samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        for key, value in calibration.stats.items():
            print(f"  {key}: {value}")

        from mlx_lm.quant.gptq import gptq_quantize

        mlx.mx.random.seed(args.seed)
        print("\nQuantizing (GPTQ)...")
        model, config["quantization"] = gptq_quantize(
            model,
            calibration.tokens,
            args.bits,
            args.group_size,
            args.fallback_bits,
            args.group_size,
        )
        config["quantization_config"] = config["quantization"]
    else:
        print("\nQuantizing (round-to-nearest)...")
        model, config = mlx.quantize_model(
            model,
            config,
            group_size=args.group_size,
            bits=args.bits,
            mode=args.mode,
            quant_predicate=predicate,
        )
    bpw = mlx.compute_bits_per_weight(model)
    print(f"  {bpw:.3f} bits/weight")

    # MLX is lazy: on the rtn path `quantize_model` only builds the graph, and
    # the arithmetic actually happens inside save() as each shard is written.
    # Timing the two separately would report 0.0s for the quantization, so time
    # the pair. Forcing an eval to split them would materialise the whole
    # quantized model at once, which is exactly the peak we want to avoid on a
    # memory-tight box.
    print(f"\nSaving to {output_dir}...")
    mlx.save(output_dir, model_path, model, tokenizer, config)
    elapsed = time.time() - t0
    print(f"  quantized and written in {elapsed:.1f}s")

    metadata = {
        "source_model": str(model_path),
        "runtime": "mlx",
        "method": args.method,
        "mode": args.mode,
        "bits": args.bits,
        "group_size": args.group_size,
        "high_bits": None if args.uniform or args.recipe else args.high_bits,
        "recipe": args.recipe,
        "bits_per_weight": round(bpw, 4),
        "total_params": plan["total_params"],
        "size_gb": round(estimate_size_gb(plan["total_params"], bpw), 3),
        "elapsed_seconds": round(elapsed, 1),  # quantize + write, see above
        "calibration": calibration.stats if calibration else None,
    }
    (output_dir / "quantization_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"\nDone — {estimate_size_gb(plan['total_params'], bpw):.2f} GiB")
    print(f"  mlx_lm.generate --model {output_dir} --prompt 'def fib(n):'")
    print(f"\nNote: {MTP_NOTE}")
