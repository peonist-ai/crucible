"""Side-by-side comparison of test bench results."""

from __future__ import annotations

import json
import math
from pathlib import Path


def compare(baseline: dict, compressed: dict) -> dict:
    """Compare two test bench result dicts.

    Aligns on matching tasks, computes deltas and percentage change.
    Warns when sample sizes are too small for statistical significance.
    """
    b_results = baseline["results"]
    c_results = compressed["results"]

    tasks = {}
    for task_name in b_results:
        if task_name not in c_results:
            continue

        b = b_results[task_name]
        c = c_results[task_name]
        b_score = b["score"]
        c_score = c["score"]
        b_n = b["total"]
        c_n = c["total"]
        delta = c_score - b_score
        delta_pct = (delta / b_score * 100) if b_score != 0 else 0
        # Recovery: the compressed score as a percentage of the baseline's. The
        # unit compression papers and llm-compressor's own REAP results report,
        # so quoting it makes our numbers directly comparable to theirs instead
        # of only internally consistent.
        recovery = (c_score / b_score * 100) if b_score != 0 else None

        # Standard error of difference between two proportions
        se = _se_diff(b_score, b_n, c_score, c_n)
        significant = abs(delta) > 1.96 * se if se > 0 else False

        tasks[task_name] = {
            "baseline_score": b_score,
            "baseline_n": b_n,
            "compressed_score": c_score,
            "compressed_n": c_n,
            "delta": delta,
            "delta_pct": delta_pct,
            "recovery": recovery,
            "se": se,
            "significant": significant,
        }

    deltas = [t["delta"] for t in tasks.values()]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    recoveries = [
        t["recovery"] for t in tasks.values() if t["recovery"] is not None
    ]
    avg_recovery = sum(recoveries) / len(recoveries) if recoveries else None

    return {
        "baseline_model": baseline["model"],
        "compressed_model": compressed["model"],
        "baseline_timestamp": baseline.get("timestamp", "?"),
        "compressed_timestamp": compressed.get("timestamp", "?"),
        "tasks": tasks,
        "avg_delta": avg_delta,
        "avg_recovery": avg_recovery,
        "num_tasks": len(tasks),
    }


def _se_diff(p1: float, n1: int, p2: float, n2: int) -> float:
    """Standard error of difference between two proportions."""
    if n1 == 0 or n2 == 0:
        return 0.0
    var1 = p1 * (1 - p1) / n1
    var2 = p2 * (1 - p2) / n2
    return math.sqrt(var1 + var2)


def print_comparison(comp: dict) -> None:
    """Print a formatted comparison table."""
    print(f"\n  Baseline:   {comp['baseline_model']}  ({comp['baseline_timestamp']})")
    print(f"  Compressed: {comp['compressed_model']}  ({comp['compressed_timestamp']})")
    print()

    header = (
        f"  {'Task':<18} {'Baseline':>12} {'Compressed':>12} {'Delta':>8} "
        f"{'Recovery':>9} {'Sig':>5}"
    )
    print(header)
    print(f"  {'-'*67}")

    for task_name, d in comp["tasks"].items():
        b_str = f"{d['baseline_score']:.1%} ({d['baseline_n']})"
        c_str = f"{d['compressed_score']:.1%} ({d['compressed_n']})"
        sign = "+" if d["delta"] >= 0 else ""
        delta_str = f"{sign}{d['delta']:.1%}"
        rec_str = "—" if d["recovery"] is None else f"{d['recovery']:.2f}%"
        sig_str = "  *" if d["significant"] else "  -"
        if d["baseline_n"] < 100 or d["compressed_n"] < 100:
            sig_str = "  ?"  # too few samples to judge
        print(
            f"  {task_name:<18} {b_str:>12} {c_str:>12} {delta_str:>8} "
            f"{rec_str:>9} {sig_str:>5}"
        )

    print(f"  {'-'*67}")
    avg = comp["avg_delta"]
    avg_delta_str = f"{'+' if avg >= 0 else ''}{avg:.1%}"
    avg_rec = comp.get("avg_recovery")
    avg_rec_str = "—" if avg_rec is None else f"{avg_rec:.2f}%"
    print(
        f"  {'average':<18} {'':>12} {'':>12} {avg_delta_str:>8} "
        f"{avg_rec_str:>9}"
    )

    # Warn about small samples
    small = [t for t, d in comp["tasks"].items()
             if d["baseline_n"] < 100 or d["compressed_n"] < 100]
    if small:
        print(f"\n  Note: {', '.join(small)} have <100 samples — deltas may be noise.")
        print("  Sig column: * = p<0.05, - = not significant, ? = too few samples")


def markdown_comparison(comp: dict) -> str:
    """Render the comparison as a markdown table.

    For model cards and READMEs. Recovery is the column that makes a compression
    result legible next to anyone else's — an absolute score says nothing without
    the baseline it came from.
    """
    lines = [
        f"| task | {comp['baseline_model']} | {comp['compressed_model']} "
        f"| delta | recovery |",
        "|---|---|---|---|---|",
    ]
    for task_name, d in comp["tasks"].items():
        sign = "+" if d["delta"] >= 0 else ""
        rec = "—" if d["recovery"] is None else f"{d['recovery']:.2f}%"
        lines.append(
            f"| {task_name} | {d['baseline_score']:.1%} ({d['baseline_n']}) "
            f"| {d['compressed_score']:.1%} ({d['compressed_n']}) "
            f"| {sign}{d['delta']:.1%} | {rec} |"
        )
    avg = comp["avg_delta"]
    avg_rec = comp.get("avg_recovery")
    avg_rec_str = "—" if avg_rec is None else f"{avg_rec:.2f}%"
    lines.append(
        f"| **average** | | | {'+' if avg >= 0 else ''}{avg:.1%} "
        f"| **{avg_rec_str}** |"
    )
    return "\n".join(lines)


def load_results(path: str | Path) -> dict:
    """Load test bench results from JSON."""
    with open(path) as f:
        return json.load(f)
