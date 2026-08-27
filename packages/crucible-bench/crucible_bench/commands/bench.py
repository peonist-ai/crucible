"""crucible-bench run — benchmark a served model over an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import sys

NAME = "run"
HELP = "Benchmark a served model via OpenAI-compatible API"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default="http://localhost:8091",
                        help="API base URL (default: http://localhost:8091)")
    parser.add_argument("--model", required=True,
                        help="Label for this model variant (e.g. baseline, reap-37pct)")
    parser.add_argument("--suite",
                        choices=["quick", "coding", "coding_plus",
                                 "regression", "agentic", "instruct", "full"],
                        default=None,
                        help="Benchmark suite. `regression` is the cheap "
                             "compression gate; `agentic` is BFCL multi-turn and "
                             "is much slower — several requests per turn, 4-6 "
                             "turns per entry.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Override suite with specific tasks")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override every task's own token budget. Omit to use each "
                             "task's declared default (bigcodebench and "
                             "gpqa_diamond 4096, everything else 2048), which is "
                             "almost always correct. Note this is per *request*: "
                             "a multi-turn entry sends many.")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["minimal", "low", "medium", "high", "xhigh"],
                        help="Sent as `reasoning_effort`. Newer servers use this "
                             "to trade thinking depth for latency; measured on a "
                             "Qwen 3.8 seat, `low` halved wall time on GPQA-hard "
                             "items. Omit to take the server's default — which is "
                             "not necessarily cheap: that seat defaults to `high`.")
    parser.add_argument("--extra-body", default=None,
                        help="JSON merged into every request body, for whatever "
                             "your server spells differently (e.g. "
                             "'{\"thinking_token_budget\": 2048}'). The harness "
                             "does not guess your server's dialect.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap problems per task (for quick testing)")
    parser.add_argument("-o", "--output", default="results",
                        help="Results output dir (default: results)")
    parser.add_argument("--sandbox",
                        choices=["auto", "docker", "podman", "none"],
                        default="auto",
                        help="How to run model-generated code. auto: "
                             "containerize if podman/docker is present, else "
                             "run locally with a warning. docker/podman: "
                             "require that runtime. none: run locally.")
    parser.add_argument("--sandbox-image", default="python:3.12-slim",
                        help="Image for sandboxed execution. BigCodeBench "
                             "needs one carrying pandas/numpy/flask/etc.")


def _resolve_extra_body(args) -> dict:
    """Assemble the per-request generation options, CLI-explicit only."""
    import json

    extra: dict = {}
    if args.extra_body:
        parsed = json.loads(args.extra_body)
        if not isinstance(parsed, dict):
            raise ValueError("--extra-body must be a JSON object")
        extra.update(parsed)
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    return extra


def run(args) -> None:
    from crucible_bench.testbench import (
        SUITES,
        SUPPORTED_TASKS,
        run_bench,
        set_sandbox,
    )

    runtime = set_sandbox(args.sandbox, args.sandbox_image)
    print(f"Code execution: {runtime or 'local (no isolation)'}")

    # Resolve tasks from suite or explicit list
    if args.tasks:
        tasks = args.tasks
    elif args.suite:
        tasks = SUITES[args.suite]
    else:
        tasks = ["humaneval"]
        print("No --suite or --tasks specified, defaulting to humaneval")

    for t in tasks:
        if t not in SUPPORTED_TASKS:
            print(f"Unknown task: {t}")
            print(f"Supported: {SUPPORTED_TASKS}")
            sys.exit(1)

    run_bench(
        url=args.url,
        model=args.model,
        tasks=tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        limit=args.limit,
        output=args.output,
        extra_body=_resolve_extra_body(args),
    )
