"""Benchmark runner wrapping EleutherAI lm-evaluation-harness.

Runs standardized benchmarks and saves structured results.
Requires: pip install lm-eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Benchmark suites from CLAUDE.md evaluation protocol.
# Primary metrics are what we optimize for (coding agents).
# Secondary are sanity checks (should not crater).
#
# Tasks prefixed with "ext:" are run via external tools (LiveCodeBench, BFCL),
# not lm-eval. The runner handles this split automatically.
SUITES: dict[str, list[str]] = {
    # Fast smoke test (~5-10 min)
    "quick": [
        "hellaswag",
        "arc_easy",
    ],
    # Primary — Coding
    "coding": [
        "humaneval",
        "mbpp",
        "ext:livecodebench",
    ],
    # Primary — Reasoning (code-adjacent)
    "reasoning": [
        "gsm8k",
        "hendrycks_math",
        "arc_challenge",
    ],
    # Primary — Instruction following + tool use
    "instruct": [
        "ifeval",
        "ext:bfcl",
    ],
    # Secondary — General knowledge (sanity check, not optimization target)
    "general": [
        "mmlu",
        "hellaswag",
    ],
    # Full suite — everything from CLAUDE.md
    "full": [
        # Primary — Coding
        "humaneval",
        "mbpp",
        "ext:livecodebench",
        # Primary — Tool use & instruction following
        "ifeval",
        "ext:bfcl",
        # Primary — Reasoning
        "gsm8k",
        "hendrycks_math",
        "arc_challenge",
        # Secondary — General
        "mmlu",
        "hellaswag",
    ],
}


def run_benchmark(
    model_path: str,
    tasks: list[str] | None = None,
    suite: str = "quick",
    batch_size: int = 4,
    num_fewshot: int | None = None,
    output_dir: str | None = None,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> dict:
    """Run benchmarks on a model.

    Handles both lm-eval tasks and external tools (LiveCodeBench, BFCL).
    Tasks prefixed with "ext:" are dispatched to their respective runners.

    Args:
        model_path: HuggingFace model ID or local path.
        tasks: specific task names. Overrides suite if provided.
        suite: preset suite name ("quick", "coding", "instruct", "full").
        batch_size: eval batch size.
        num_fewshot: override default few-shot count per task.
        output_dir: save results JSON here.
        device: device for evaluation.
        dtype: model dtype.

    Returns:
        dict with task results, metadata, and timing.
    """
    task_list = tasks if tasks else SUITES.get(suite, SUITES["quick"])

    # Split into lm-eval tasks and external tasks
    lm_eval_tasks = [t for t in task_list if not t.startswith("ext:")]
    ext_tasks = [t for t in task_list if t.startswith("ext:")]

    print(f"  Model: {model_path}")
    if lm_eval_tasks:
        print(f"  lm-eval tasks: {', '.join(lm_eval_tasks)}")
    if ext_tasks:
        print(f"  External tasks: {', '.join(ext_tasks)}")
    print(f"  Batch size: {batch_size}")

    scores = {}
    t0 = time.time()

    # --- lm-eval tasks ---
    if lm_eval_tasks:
        import lm_eval

        model_args = f"pretrained={model_path},dtype={dtype}"
        if device != "auto":
            model_args += f",device={device}"

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=lm_eval_tasks,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            confirm_run_unsafe_code=True,
        )

        for task_name, task_result in results.get("results", {}).items():
            score = None
            for key in [
                "acc,none", "acc_norm,none",
                "exact_match,none", "pass@1,none",
            ]:
                if key in task_result:
                    score = task_result[key]
                    break
            if score is None:
                for k, v in task_result.items():
                    if isinstance(v, (int, float)) and k != "alias":
                        score = v
                        break

            scores[task_name] = {
                "score": score,
                "raw": {
                    k: v for k, v in task_result.items() if k != "alias"
                },
            }

    # --- External tasks ---
    for ext_task in ext_tasks:
        task_name = ext_task.removeprefix("ext:")
        try:
            ext_result = _run_external_task(
                task_name, model_path, device, dtype
            )
            scores[task_name] = ext_result
        except Exception as e:
            print(f"  WARNING: {task_name} failed: {e}")
            scores[task_name] = {"score": None, "error": str(e)}

    elapsed = time.time() - t0

    output = {
        "model": model_path,
        "tasks": task_list,
        "scores": scores,
        "elapsed_seconds": elapsed,
        "batch_size": batch_size,
        "config": {
            "dtype": dtype,
            "device": device,
            "num_fewshot": num_fewshot,
        },
    }

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_name = model_path.rstrip("/").split("/")[-1]
        result_file = out_path / f"{model_name}_eval.json"
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to {result_file}")

    return output


def _run_external_task(
    task_name: str, model_path: str, device: str, dtype: str
) -> dict:
    """Dispatch to external benchmark tools."""
    if task_name == "livecodebench":
        return _run_livecodebench(model_path)
    if task_name == "bfcl":
        return _run_bfcl(model_path)
    raise ValueError(f"Unknown external task: {task_name}")


def _run_livecodebench(model_path: str) -> dict:
    """Run LiveCodeBench code generation evaluation.

    Requires: pip install livecodebench
    """
    import subprocess

    print("  Running LiveCodeBench...")
    result = subprocess.run(
        [
            "python", "-m", "lcb_runner.runner.main",
            "--model", model_path,
            "--scenario", "codegeneration",
            "--evaluate",
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LiveCodeBench failed:\n{result.stderr[-500:]}"
        )

    # Parse pass@1 from output
    score = _parse_pass_at_1(result.stdout)
    return {
        "score": score,
        "raw": {"stdout": result.stdout[-2000:]},
    }


def _run_bfcl(model_path: str) -> dict:
    """Run Berkeley Function Calling Leaderboard evaluation.

    Requires: pip install bfcl
    """
    import subprocess

    print("  Running BFCL...")

    # Generate responses
    gen_result = subprocess.run(
        [
            "bfcl", "generate",
            "--model", model_path,
            "--test-category", "all",
            "--backend", "vllm",
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if gen_result.returncode != 0:
        raise RuntimeError(
            f"BFCL generate failed:\n{gen_result.stderr[-500:]}"
        )

    # Evaluate responses
    eval_result = subprocess.run(
        [
            "bfcl", "evaluate",
            "--model", model_path,
            "--test-category", "all",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if eval_result.returncode != 0:
        raise RuntimeError(
            f"BFCL evaluate failed:\n{eval_result.stderr[-500:]}"
        )

    score = _parse_bfcl_score(eval_result.stdout)
    return {
        "score": score,
        "raw": {"stdout": eval_result.stdout[-2000:]},
    }


def _parse_pass_at_1(output: str) -> float | None:
    """Parse pass@1 score from LiveCodeBench output."""
    import re

    match = re.search(r"pass@1[:\s]+([0-9.]+)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _parse_bfcl_score(output: str) -> float | None:
    """Parse overall accuracy from BFCL output."""
    import re

    match = re.search(
        r"(?:overall|total|accuracy)[:\s]+([0-9.]+)", output, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    return None


def print_scores(results: dict) -> None:
    """Print a formatted score table."""
    print(f"\n{'Task':<25} {'Score':>10}")
    print("-" * 37)
    for task_name, task_data in results["scores"].items():
        score = task_data["score"]
        if score is not None:
            if isinstance(score, float):
                print(f"{task_name:<25} {score:>10.4f}")
            else:
                print(f"{task_name:<25} {score!s:>10}")
        else:
            print(f"{task_name:<25} {'N/A':>10}")
    print("-" * 37)
    elapsed = results.get("elapsed_seconds", 0)
    print(f"Total time: {elapsed:.0f}s")
