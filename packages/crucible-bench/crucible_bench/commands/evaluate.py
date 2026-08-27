"""crucible-bench eval-local — lm-evaluation-harness over local weights.

The odd one out: every other command here talks to an endpoint over HTTP,
this one loads a checkpoint into memory. That is why lm-eval (and the torch
it drags in) is the `local` extra rather than a base dependency — installing
crucible-bench to benchmark a remote server should not cost you a 2GB wheel.
"""

from __future__ import annotations

import argparse

NAME = "eval-local"
HELP = "Evaluate local weights with lm-evaluation-harness"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--suite",
        choices=["quick", "coding", "reasoning", "instruct", "general", "full"],
        default="quick",
        help="Benchmark suite (default: quick)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Override suite with specific tasks"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("-o", "--output", default="results", help="Results output dir")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device.

    Duplicated from crucible's commands/common.py rather than imported: a
    cross-package import would re-couple the two packages over twenty lines,
    and torch stays lazily imported either way.
    """
    if device_str != "auto":
        return device_str

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype_str(dtype_str: str) -> str:
    if dtype_str == "auto":
        return "bfloat16"
    return dtype_str


def run(args) -> None:
    from crucible_bench.benchmark import print_scores, run_benchmark

    dtype_str = _resolve_dtype_str(args.dtype)

    print(f"Evaluating {args.model}")
    print(f"Suite: {args.suite}")
    print()

    results = run_benchmark(
        model_path=args.model,
        tasks=args.tasks,
        suite=args.suite,
        batch_size=args.batch_size,
        output_dir=args.output,
        device=_resolve_device(args.device),
        dtype=dtype_str,
    )

    print_scores(results)
