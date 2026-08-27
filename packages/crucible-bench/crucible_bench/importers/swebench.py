"""Import a SWE-bench evaluation report.

`python -m swebench.harness.run_evaluation` writes a summary at
`<model>.<run_id>.json` alongside per-instance reports under
`logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json`. We read the
summary; the per-instance logs stay where the harness put them, and the
provenance block points at the summary so they can be found.

Summary keys we rely on:

    total_instances, submitted_instances, completed_instances,
    resolved_instances, unresolved_instances, empty_patch_instances,
    error_instances, and the matching *_ids lists.

The denominator is `total_instances`, which is what SWE-bench's own
leaderboard divides by. Scoring against `completed_instances` instead would
quietly delete every instance the harness failed to run and inflate the result
-- the same trick as dropping a benchmark's hard half.
"""

from __future__ import annotations

import json
from pathlib import Path

from crucible_bench.importers.registry import ImportedRun

DEFAULT_TASK = "swebench"

_REQUIRED = ("total_instances", "resolved_ids")


def _find_report(path: Path) -> Path:
    if path.is_file():
        return path
    reports = sorted(p for p in path.glob("*.json") if _looks_like_report(p))
    if len(reports) == 1:
        return reports[0]
    if not reports:
        raise FileNotFoundError(
            f"no SWE-bench report JSON in {path}. Point --from at the "
            f"<model>.<run_id>.json the harness wrote, or at a directory "
            f"containing exactly one."
        )
    raise ValueError(
        f"{len(reports)} report files in {path}: {[p.name for p in reports]}. "
        f"Name the one you mean with --from."
    )


def _looks_like_report(path: Path) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
    except (ValueError, OSError):
        return False
    return isinstance(data, dict) and all(k in data for k in _REQUIRED)


def import_run(path: Path, task: str | None = None) -> ImportedRun:
    report_path = _find_report(Path(path))
    with open(report_path) as f:
        report = json.load(f)

    missing = [k for k in _REQUIRED if k not in report]
    if missing:
        raise ValueError(
            f"{report_path} is missing {missing}. That is not a SWE-bench "
            f"evaluation report — importing it would invent a score."
        )

    total = int(report["total_instances"])
    if total == 0:
        raise ValueError(f"{report_path} reports zero instances; nothing to import.")

    resolved = set(report.get("resolved_ids") or [])
    unresolved = set(report.get("unresolved_ids") or [])
    empty_patch = set(report.get("empty_patch_ids") or [])
    errored = set(report.get("error_ids") or [])
    incomplete_ids = set(report.get("incomplete_ids") or [])

    # Everything the report knows about, so an instance that appears in no
    # list still shows up as unaccounted-for rather than vanishing.
    known = resolved | unresolved | empty_patch | errored | incomplete_ids
    details = []
    for instance_id in sorted(known):
        if instance_id in resolved:
            reason = None
        elif instance_id in empty_patch:
            reason = "empty_patch"
        elif instance_id in errored:
            reason = "harness_error"
        elif instance_id in incomplete_ids:
            reason = "incomplete"
        else:
            reason = "tests_failed"
        details.append({
            "task_id": instance_id,
            "passed": instance_id in resolved,
            "reason": reason,
        })

    unaccounted = total - len(known)
    incomplete = len(empty_patch | errored | incomplete_ids) + max(unaccounted, 0)

    warnings = []
    if errored or incomplete_ids:
        warnings.append(
            f"{len(errored | incomplete_ids)}/{total} instances did not complete "
            f"in the harness. They count as unresolved (SWE-bench's own rule) but "
            f"are not evidence about the model."
        )
    if empty_patch:
        warnings.append(
            f"{len(empty_patch)}/{total} instances produced an empty patch — the "
            f"agent finished without editing anything. Check the agent scaffold "
            f"and its turn limit before attributing this to the model."
        )
    if unaccounted > 0:
        warnings.append(
            f"{unaccounted}/{total} instances appear in no id list in the report. "
            f"They are scored as failures; the report may be a schema version "
            f"this importer has not seen."
        )

    provenance = {
        "tool": "swebench",
        "harness": "swebench.harness.run_evaluation",
        "source": str(report_path),
        "report_schema_version": report.get("schema_version"),
        "total_instances": total,
        "submitted_instances": report.get("submitted_instances"),
        "completed_instances": report.get("completed_instances"),
        "empty_patch_instances": report.get("empty_patch_instances"),
        "error_instances": report.get("error_instances"),
        # Per-instance logs (test output, applied patch, harness log) live
        # beside the report and are the only way to audit a single failure.
        "per_instance_logs": str(
            report_path.parent / "logs" / "run_evaluation"
        ),
    }

    return ImportedRun(
        task=task or DEFAULT_TASK,
        score=len(resolved) / total,
        passed=len(resolved),
        total=total,
        details=details,
        provenance=provenance,
        incomplete=incomplete,
        warnings=warnings,
    )
