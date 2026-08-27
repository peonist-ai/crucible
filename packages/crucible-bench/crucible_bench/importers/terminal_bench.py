"""Import a Harbor run — Terminal-Bench 2.0 and anything else Harbor drives.

Harbor writes `results.json` at the run root: a `JobResult` carrying
`trial_results`, one `TrialResult` per task attempt. Per-trial pass/fail lives
in `verifier_result.rewards`, a `dict[str, float | int]`.

Schema (harbor `src/harbor/models/{job,trial,verifier}/result.py`):

    JobResult    { id, started_at, finished_at, n_total_trials, stats,
                   trial_results: [TrialResult] }
    TrialResult  { task_name, trial_name, agent_info, verifier_result,
                   exception_info, started_at, finished_at, ... }
    AgentInfo    { name, version, model_info: { name, provider } }
    VerifierResult { rewards: dict[str, float|int] | None }

We read the trial list rather than `stats.pass_at_k` so the per-task detail
survives into the result file. The fail-set overlap method needs task-level
outcomes, not a rate -- at 89 tasks a marginal rate cannot tell a compression
delta from noise, but a paired per-task comparison can.
"""

from __future__ import annotations

import json
from pathlib import Path

from crucible_bench.importers.registry import ImportedRun

DEFAULT_TASK = "terminal_bench"

# Harbor's convention for a solved task. Rewards are 0/1 in Terminal-Bench;
# a partial-credit benchmark would need its own threshold, which is why this
# is named rather than inlined.
SOLVED_REWARD = 1.0


def _find_results_json(path: Path) -> Path:
    """Accept either the run directory or the results.json inside it."""
    if path.is_file():
        return path
    candidate = path / "results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"no results.json at {path}. Point --from at a Harbor run directory "
        f"(the one containing results.json and run.log) or at the file itself."
    )


def _reward(rewards: dict | None, task_name: str) -> float | None:
    """The one number that says whether this trial solved the task.

    Returns None when there is no verdict. Raises when there are several
    rewards and no way to tell which one means "solved" -- picking the first
    key, or the max, would be a guess presented as a measurement.
    """
    if not rewards:
        return None
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    raise ValueError(
        f"trial {task_name!r} reported several rewards {sorted(rewards)} and "
        f"none named 'reward'. Which one means solved is a judgement call this "
        f"importer will not make for you -- open the run and decide, then add "
        f"the rule here."
    )


def import_run(path: Path, task: str | None = None) -> ImportedRun:
    results_path = _find_results_json(Path(path))
    with open(results_path) as f:
        job = json.load(f)

    trials = job.get("trial_results") or []
    if not trials:
        raise ValueError(
            f"{results_path} has no trial_results. An empty Harbor run scores "
            f"0% and looks exactly like a model that solved nothing."
        )

    details: list[dict] = []
    passed = 0
    incomplete = 0
    agent_names: set[str] = set()
    model_names: set[str] = set()

    for trial in trials:
        name = trial.get("task_name", "?")
        exc = trial.get("exception_info")
        verifier = trial.get("verifier_result") or {}
        reward = _reward(verifier.get("rewards"), name)

        if reward is None:
            incomplete += 1
            solved = False
        else:
            solved = reward >= SOLVED_REWARD
            if solved:
                passed += 1

        agent = trial.get("agent_info") or {}
        agent_names.add(agent.get("name", "?"))
        model_info = agent.get("model_info") or {}
        if model_info.get("name"):
            model_names.add(model_info["name"])

        details.append({
            "task_id": name,
            "trial": trial.get("trial_name"),
            "passed": solved,
            "reward": reward,
            # An exception is infrastructure, not the model failing the task.
            # Scored as a failure because Harbor counts it that way, recorded
            # so a broken run cannot masquerade as a regression.
            "error": (exc or {}).get("exception_type"),
            "error_message": (exc or {}).get("exception_message"),
        })

    total = len(trials)
    stats = job.get("stats") or {}
    warnings = []
    if incomplete:
        warnings.append(
            f"{incomplete}/{total} trials produced no verdict (container or "
            f"agent errors). They are scored as failures — check `error` in "
            f"the details before reading this as model quality."
        )

    provenance = {
        "tool": "terminal-bench",
        "harness": "harbor",
        "source": str(results_path),
        "job_id": job.get("id"),
        "agents": sorted(agent_names),
        "models_reported_by_tool": sorted(model_names),
        "n_total_trials": job.get("n_total_trials"),
        "n_errored_trials": stats.get("n_errored_trials"),
        "n_cancelled_trials": stats.get("n_cancelled_trials"),
        "n_retries": stats.get("n_retries"),
        "cost_usd": stats.get("cost_usd"),
        # Harbor's own pass@k, kept alongside ours so a disagreement is visible
        # rather than silently resolved in our favour.
        "harbor_pass_at_k": {
            eval_name: s.get("pass_at_k")
            for eval_name, s in (stats.get("evals") or {}).items()
        },
    }

    return ImportedRun(
        task=task or DEFAULT_TASK,
        score=passed / total,
        passed=passed,
        total=total,
        details=details,
        provenance=provenance,
        incomplete=incomplete,
        elapsed=_elapsed(job.get("started_at"), job.get("finished_at")),
        timestamp=job.get("finished_at") or job.get("started_at"),
        warnings=warnings,
    )


def _elapsed(started: str | None, finished: str | None) -> float | None:
    if not started or not finished:
        return None
    from datetime import datetime

    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return round((t1 - t0).total_seconds(), 1)
