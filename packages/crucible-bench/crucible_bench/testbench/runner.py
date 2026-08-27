"""Run a set of benchmarks against one endpoint and record the results.

Results are written as one JSON file per run — model label, config, per-task
scores, and every per-problem record. Comparing runs is `crucible-bench compare`'s
job; this file's only obligation is to never lose what it measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from crucible_bench.testbench.api import api_get
from crucible_bench.testbench.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    SANDBOX_MODES,
    set_sandbox,
)
from crucible_bench.testbench.suites import SUITES, SUPPORTED_TASKS, TASKS


def _count_truncated(results: list[dict]) -> int | None:
    """How many problems stopped because they hit the token cap.

    Returns None when the task doesn't record `finish_reason` — "not measured"
    and "zero" must not look alike, because a run that silently truncates scores
    like a model that silently fails.
    """
    reported = [r for r in results if "finish_reason" in r]
    if not reported:
        return None
    return sum(1 for r in reported if r["finish_reason"] == "length")


def _count_errors(results: list[dict]) -> int | None:
    """How many problems never got an answer out of the server at all.

    Transport failures survive retries sometimes -- a server down for the whole
    run, a network that never comes back. Those problems are scored wrong
    because there is nothing else to do with them, but they are not evidence
    about the model and must never be read as such.
    """
    reported = [r for r in results if "finish_reason" in r]
    if not reported:
        return None
    return sum(1 for r in reported if r["finish_reason"] == "error")


def _count_unparsed(results: list[dict]) -> int | None:
    """How many answers no extraction pattern could read.

    Returns None when the task doesn't record `extraction_tier`. These are
    scored wrong, which is correct -- but a score with a large unparsed count
    is measuring the answer *format* as much as the answer, and that belongs
    next to the number rather than buried in the details.
    """
    reported = [r for r in results if "extraction_tier" in r]
    if not reported:
        return None
    return sum(1 for r in reported if r["extraction_tier"] == "none")


def _check_ignored_params(url: str, extra_body: dict | None) -> None:
    """Ask the server whether it will actually honour what we are sending.

    Some servers expose a /health that names which request parameters they
    accept-but-ignore. Sending one of those and then recording it in the
    results file states a generation setting that never reached the model —
    which is how a run at the server's default effort came to be filed as a
    run at a budget we chose.

    Best effort by design: no /health, or one without these fields, is not an
    error. A server that does tell us is one we should listen to.
    """
    if not extra_body:
        return
    try:
        health = api_get(f"{url}/health")
    except Exception:
        return
    if not isinstance(health, dict):
        return

    ignored = set(health.get("accepted_but_ignored") or [])
    supported = set(health.get("supported") or [])
    for key in extra_body:
        if key in ignored:
            print(f"  WARNING: the server lists {key!r} as accepted-but-ignored. "
                  f"It will not affect generation, and recording it would "
                  f"misdescribe this run.")
        elif supported and key not in supported:
            print(f"  WARNING: {key!r} is not in the server's supported list "
                  f"{sorted(supported)}. It may be silently dropped — verify it "
                  f"changes behaviour before trusting a score labelled with it.")


# How often a long task rewrites its partial results.
CHECKPOINT_SECONDS = 30


def _score_task(results, max_tokens, spec, extra_body) -> dict:
    """Score a record list. Shared so a mid-task checkpoint and the final
    write cannot disagree about what the same records mean."""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    return {
        "score": passed / total if total > 0 else 0,
        "passed": passed,
        "total": total,
        "max_tokens": max_tokens,
        "reasoning": spec.reasoning,
        # Exactly what was merged into every request body for this task.
        # Recorded rather than inferred: the previous field named a thinking
        # budget the server never honoured.
        "extra_body": extra_body or {},
        # None means the task doesn't report finish_reason, not "none hit the
        # cap" -- the two are very different when a score looks wrong.
        "truncated": _count_truncated(results),
        # Requests that never completed. Not model failures.
        "errored": _count_errors(results),
        # Likewise: None means the task doesn't extract an answer at all, not
        # that every answer parsed.
        "unparsed": _count_unparsed(results),
        "details": results,
    }


def _build_result(model, run_id, timestamp, url, config, task_results,
                  task_elapsed, tasks_requested, complete, in_flight=None):
    """Render the result file from whatever has finished so far."""
    return {
        "model": model,
        "run_id": run_id,
        "timestamp": timestamp,
        "url": url,
        # A partial file must never be mistakable for a finished one. `complete`
        # is False until every requested task has been scored, and
        # `tasks_requested` says what a finished run would have contained.
        "complete": complete,
        "tasks_requested": list(tasks_requested),
        # Named when a task was still running at the last write: its scores are
        # over however many problems had finished, not the whole task.
        "in_flight": in_flight,
        "config": config,
        "results": {
            task: {k: v for k, v in res.items() if k != "details"}
            for task, res in task_results.items()
        },
        "details": {task: res["details"] for task, res in task_results.items()},
        "elapsed": task_elapsed,
    }


def _save(out_file: Path, result: dict) -> None:
    """Write the result file atomically.

    Written after every task, not once at the end. A killed run used to
    discard everything it had measured -- 48 minutes of HumanEval+ and 107
    MBPP+ problems went in the bin on 2026-08-22 because the process was
    stopped during the third task. The scores survived only because someone
    happened to be teeing stdout to a log; the per-problem detail did not.

    Atomic because the alternative is a truncated JSON file where a complete
    earlier one used to be: an interrupted write during a long sweep would
    destroy the very results this is meant to protect.
    """
    tmp = out_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    tmp.replace(out_file)


def run_bench(
    url: str,
    model: str,
    tasks: list[str],
    max_tokens: int | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    limit: int | None = None,
    output: str = "results",
    extra_body: dict | None = None,
) -> dict:
    """Run benchmarks and return structured results.

    Args:
        url: OpenAI-compatible API base URL.
        model: Label for this model variant (e.g. "baseline", "reap-37pct-routing").
        tasks: List of benchmark task names to run.
        max_tokens: Override every task's own budget. None (the default) keeps each
            task's declared TaskSpec.max_tokens, which is almost always what you want.
        temperature: Sampling temperature.
        seed: Random seed for MC task shuffling.
        limit: Cap number of problems per task (for quick testing).
        output: Directory to save result JSON.
        extra_body: Server-specific generation options merged into every
            request (e.g. {"reasoning_effort": "low"}). Passed through
            verbatim — the harness does not know your server's dialect.

    Returns:
        Structured result dict with metadata, scores, details, and timing.
    """
    # Verify server is reachable
    api_get(f"{url}/v1/models")

    # Run identity is fixed before the first request so every incremental save
    # lands in the same file.
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"{model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}.json"
    config = {
        # None means "each task used its own declared budget"; see each task's
        # results[task]["max_tokens"] for what was actually sent.
        "max_tokens_override": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "limit": limit,
        "extra_body": extra_body or {},
    }

    _check_ignored_params(url, extra_body)

    task_results = {}
    task_elapsed = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  {task}")
        print(f"{'='*60}")

        spec = TASKS[task]
        t0 = time.monotonic()
        # The task's own budget wins unless explicitly overridden: a 512-token cap
        # truncates BigCodeBench solutions mid-function and scores that as a model
        # failure. Record the EFFECTIVE value per task -- reporting the CLI arg here
        # instead once sent an investigation down the wrong path for an hour.
        effective_max_tokens = spec.max_tokens if max_tokens is None else max_tokens
        last_save = [time.monotonic()]

        def checkpoint(partial, _task=task, _spec=spec,
                       _budget=effective_max_tokens):
            """Persist mid-task, at most every CHECKPOINT_SECONDS.

            Throttled because the alternative is rewriting a growing JSON file
            after every single problem — on a 400-case task with full response
            text that is real I/O, and it buys nothing over a 30-second window.
            """
            if time.monotonic() - last_save[0] < CHECKPOINT_SECONDS:
                return
            last_save[0] = time.monotonic()
            snapshot = dict(task_results)
            snapshot[_task] = _score_task(partial, _budget, _spec, extra_body)
            _save(out_file, _build_result(model, run_id, timestamp, url, config,
                                          snapshot, task_elapsed, tasks,
                                          complete=False, in_flight=_task))

        results = spec.run(url, effective_max_tokens, temperature,
                           limit, seed, extra_body, checkpoint)
        elapsed = time.monotonic() - t0
        task_elapsed[task] = round(elapsed, 1)

        task_results[task] = _score_task(results, effective_max_tokens, spec,
                                         extra_body)
        passed = task_results[task]["passed"]
        total = task_results[task]["total"]
        score = task_results[task]["score"]
        truncated = task_results[task]["truncated"]
        unparsed = task_results[task]["unparsed"]
        errored = task_results[task]["errored"]

        print(f"\n  {task}: {passed}/{total} = {score:.1%}  ({elapsed:.0f}s)")
        if truncated:
            print(f"  WARNING: {truncated}/{total} hit the {effective_max_tokens}-token cap. "
                  f"Those are budget failures, not model failures -- raise max_tokens "
                  f"before reading this score.")
        if errored:
            print(f"  WARNING: {errored}/{total} requests never completed even after "
                  f"retries (server restart, or down). They are scored wrong because "
                  f"there is nothing else to do with them, but they are NOT evidence "
                  f"about the model -- subtract them before quoting this score.")
        if unparsed:
            print(f"  NOTE: {unparsed}/{total} responses had no readable answer and were "
                  f"scored wrong. Check `extraction_tier` in the details before "
                  f"attributing this score to knowledge rather than formatting.")

        # Persist before starting the next task, so an interruption costs at
        # most the task in flight rather than the whole sweep.
        _save(out_file, _build_result(model, run_id, timestamp, url, config,
                                      task_results, task_elapsed, tasks,
                                      complete=False))
        print(f"  saved → {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task, res in task_results.items():
        t = task_elapsed.get(task, 0)
        print(f"  {task:<20} {res['score']:.1%} ({res['passed']}/{res['total']})  [{t:.0f}s]")

    result = _build_result(model, run_id, timestamp, url, config, task_results,
                           task_elapsed, tasks, complete=True)
    _save(out_file, result)
    print(f"\n  Results saved to {out_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        prog="python -m crucible_bench.testbench",
        description="Crucible test bench",
    )
    parser.add_argument("--url", default="http://localhost:8091")
    parser.add_argument("--model", required=True, help="Label for this model variant")
    parser.add_argument("--suite", choices=list(SUITES.keys()), default=None,
                        help="Benchmark suite (quick/coding/full)")
    parser.add_argument("--tasks", nargs="+", default=None, choices=SUPPORTED_TASKS,
                        help="Override suite with specific tasks")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override every task's own token budget. Omit to use each "
                             "task's declared default, which is almost always correct.")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sandbox", choices=list(SANDBOX_MODES), default="auto",
        help="How to run model-generated code. auto: containerize if podman or "
             "docker is present, else run locally with a warning. docker/podman: "
             "require that runtime. none: run locally, no warning.",
    )
    parser.add_argument(
        "--sandbox-image", default=DEFAULT_SANDBOX_IMAGE,
        help=f"Container image for sandboxed execution (default: "
             f"{DEFAULT_SANDBOX_IMAGE}). BigCodeBench needs an image carrying "
             f"pandas/numpy/flask/etc.",
    )
    args = parser.parse_args()

    set_sandbox(args.sandbox, args.sandbox_image)

    # Resolve tasks from suite or explicit list
    if args.tasks:
        tasks = args.tasks
    elif args.suite:
        tasks = SUITES[args.suite]
    else:
        tasks = ["humaneval"]

    try:
        run_bench(
            url=args.url,
            model=args.model,
            tasks=tasks,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            limit=args.limit,
            output=args.output,
            extra_body=({"reasoning_effort": args.reasoning_effort}
                        if args.reasoning_effort else None),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
