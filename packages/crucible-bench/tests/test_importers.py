"""Importers for benchmarks we deliberately don't run ourselves.

Fixtures here mirror the real upstream schemas:

  - Harbor `results.json` = `JobResult` from
    harbor/src/harbor/models/job/result.py, with `trial_results` of
    `TrialResult` and per-trial `verifier_result.rewards`.
  - SWE-bench `<model>.<run_id>.json` from `swebench.harness.run_evaluation`.

The theme of every test: an importer that cannot tell what happened must say
so, not write a zero. A silently-imported 0% is indistinguishable from a
model that got catastrophically worse, and that is the exact confusion these
benchmarks exist to resolve.
"""

import json

import pytest

from crucible_bench.importers import SUPPORTED_TOOLS, get_importer
from crucible_bench.importers.swebench import import_run as import_swebench
from crucible_bench.importers.terminal_bench import import_run as import_tb


def _trial(task_name, reward=None, exception=None, agent="terminus-2",
           model="hosted_vllm/qwen36-reap-20pct"):
    verifier = None if reward is None else {"rewards": {"reward": reward}}
    return {
        "task_name": task_name,
        "trial_name": f"{task_name}.1",
        "agent_info": {
            "name": agent,
            "version": "0.4.1",
            "model_info": {"name": model, "provider": "hosted_vllm"},
        },
        "verifier_result": verifier,
        "exception_info": (
            None if exception is None
            else {
                "exception_type": exception,
                "exception_message": "boom",
                "exception_traceback": "...",
                "occurred_at": "2026-08-22T10:00:00",
            }
        ),
    }


def _harbor_run(tmp_path, trials, **job):
    payload = {
        "id": "0f9a2c1e-0000-0000-0000-000000000001",
        "started_at": "2026-08-22T10:00:00",
        "finished_at": "2026-08-22T11:30:00",
        "n_total_trials": len(trials),
        "stats": {
            "n_completed_trials": len(trials),
            "n_errored_trials": 0,
            "n_retries": 0,
            "cost_usd": None,
            "evals": {"terminal-bench": {"n_trials": len(trials),
                                         "pass_at_k": {"1": 0.5}}},
        },
        "trial_results": trials,
    }
    payload.update(job)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(payload))
    return run_dir


class TestRegistry:
    def test_known_tools(self):
        assert SUPPORTED_TOOLS == ["terminal-bench", "swebench"]

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="no importer"):
            get_importer("nope")


class TestTerminalBench:
    def test_scores_from_per_trial_rewards(self, tmp_path):
        run = _harbor_run(tmp_path, [
            _trial("hello-world", reward=1.0),
            _trial("dna-assembly", reward=0.0),
            _trial("pytorch-model-cli", reward=1.0),
            _trial("adaptive-rejection-sampler", reward=0.0),
        ])
        result = import_tb(run, None)

        assert result.task == "terminal_bench"
        assert (result.passed, result.total) == (2, 4)
        assert result.score == 0.5
        assert result.incomplete == 0
        assert result.warnings == []

    def test_accepts_the_results_file_directly(self, tmp_path):
        run = _harbor_run(tmp_path, [_trial("hello-world", reward=1.0)])
        assert import_tb(run / "results.json", None).passed == 1

    def test_per_task_detail_survives(self, tmp_path):
        # The whole reason we read trial_results instead of stats.pass_at_k:
        # paired per-task comparison is the only thing with enough power to
        # separate a compression delta from noise at this sample size.
        run = _harbor_run(tmp_path, [
            _trial("hello-world", reward=1.0),
            _trial("dna-assembly", reward=0.0),
        ])
        result = import_tb(run, None)
        by_id = {d["task_id"]: d for d in result.details}
        assert by_id["hello-world"]["passed"] is True
        assert by_id["dna-assembly"]["passed"] is False

    def test_missing_verdict_counts_as_incomplete_and_warns(self, tmp_path):
        run = _harbor_run(tmp_path, [
            _trial("hello-world", reward=1.0),
            _trial("broken-container", exception="ContainerBuildError"),
        ])
        result = import_tb(run, None)

        assert result.passed == 1
        assert result.total == 2          # still in the denominator
        assert result.incomplete == 1
        assert any("no verdict" in w for w in result.warnings)
        broken = [d for d in result.details if d["task_id"] == "broken-container"][0]
        assert broken["passed"] is False
        assert broken["error"] == "ContainerBuildError"

    def test_ambiguous_rewards_raise_rather_than_guess(self, tmp_path):
        trial = _trial("multi-metric")
        trial["verifier_result"] = {"rewards": {"tests_passed": 1.0, "style": 0.0}}
        run = _harbor_run(tmp_path, [trial])
        with pytest.raises(ValueError, match="several rewards"):
            import_tb(run, None)

    def test_single_unnamed_reward_is_used(self, tmp_path):
        trial = _trial("solo-metric")
        trial["verifier_result"] = {"rewards": {"resolved": 1.0}}
        run = _harbor_run(tmp_path, [trial])
        assert import_tb(run, None).passed == 1

    def test_empty_run_raises(self, tmp_path):
        run = _harbor_run(tmp_path, [])
        with pytest.raises(ValueError, match="no trial_results"):
            import_tb(run, None)

    def test_missing_results_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no results.json"):
            import_tb(tmp_path, None)

    def test_provenance_records_what_we_did_not_control(self, tmp_path):
        run = _harbor_run(tmp_path, [_trial("hello-world", reward=1.0)])
        p = import_tb(run, None).provenance

        assert p["tool"] == "terminal-bench"
        assert p["harness"] == "harbor"
        assert p["agents"] == ["terminus-2"]
        assert p["models_reported_by_tool"] == ["hosted_vllm/qwen36-reap-20pct"]
        # Harbor's own rate is kept so a disagreement with ours is visible.
        assert p["harbor_pass_at_k"] == {"terminal-bench": {"1": 0.5}}

    def test_elapsed_from_job_timing(self, tmp_path):
        run = _harbor_run(tmp_path, [_trial("hello-world", reward=1.0)])
        assert import_tb(run, None).elapsed == 5400.0


def _swebench_report(tmp_path, **overrides):
    report = {
        "total_instances": 50,
        "submitted_instances": 50,
        "completed_instances": 48,
        "resolved_instances": 20,
        "unresolved_instances": 28,
        "empty_patch_instances": 1,
        "error_instances": 1,
        "resolved_ids": [f"astropy__astropy-{i}" for i in range(20)],
        "unresolved_ids": [f"django__django-{i}" for i in range(28)],
        "empty_patch_ids": ["sympy__sympy-1"],
        "error_ids": ["scikit-learn__scikit-learn-1"],
        "incomplete_ids": [],
        "schema_version": 2,
    }
    report.update(overrides)
    path = tmp_path / "qwen36-reap-20pct.run1.json"
    path.write_text(json.dumps(report))
    return path


class TestSweBench:
    def test_scores_against_total_instances(self, tmp_path):
        # Not completed_instances: dividing by what the harness managed to run
        # deletes its failures from the denominator and inflates the score.
        result = import_swebench(_swebench_report(tmp_path), None)
        assert (result.passed, result.total) == (20, 50)
        assert result.score == 0.4

    def test_finds_the_report_in_a_directory(self, tmp_path):
        _swebench_report(tmp_path)
        assert import_swebench(tmp_path, None).passed == 20

    def test_ambiguous_directory_raises(self, tmp_path):
        _swebench_report(tmp_path)
        second = tmp_path / "other.run2.json"
        second.write_text((tmp_path / "qwen36-reap-20pct.run1.json").read_text())
        with pytest.raises(ValueError, match="report files"):
            import_swebench(tmp_path, None)

    def test_failure_reasons_are_recorded(self, tmp_path):
        result = import_swebench(_swebench_report(tmp_path), None)
        by_id = {d["task_id"]: d for d in result.details}
        assert by_id["astropy__astropy-0"]["reason"] is None
        assert by_id["django__django-0"]["reason"] == "tests_failed"
        assert by_id["sympy__sympy-1"]["reason"] == "empty_patch"
        assert by_id["scikit-learn__scikit-learn-1"]["reason"] == "harness_error"

    def test_empty_patch_and_errors_warn(self, tmp_path):
        result = import_swebench(_swebench_report(tmp_path), None)
        assert any("empty patch" in w for w in result.warnings)
        assert any("did not complete" in w for w in result.warnings)
        assert result.incomplete == 2

    def test_instances_in_no_id_list_are_flagged(self, tmp_path):
        # A schema change that renames an id list would otherwise silently
        # convert those instances into failures.
        report = _swebench_report(tmp_path, unresolved_ids=[])
        result = import_swebench(report, None)
        assert any("no id list" in w for w in result.warnings)

    def test_non_report_json_raises(self, tmp_path):
        path = tmp_path / "preds.json"
        path.write_text(json.dumps({"instance_id": "x", "model_patch": "diff"}))
        with pytest.raises(ValueError, match="not a SWE-bench"):
            import_swebench(path, None)

    def test_zero_instances_raises(self, tmp_path):
        report = _swebench_report(tmp_path, total_instances=0, resolved_ids=[])
        with pytest.raises(ValueError, match="zero instances"):
            import_swebench(report, None)

    def test_task_name_override(self, tmp_path):
        result = import_swebench(_swebench_report(tmp_path), "swebench_verified_mini")
        assert result.task == "swebench_verified_mini"


class TestImportedRunsCompare:
    """An imported run must be diffable against a native one."""

    def test_compare_accepts_an_imported_result(self, tmp_path):
        from crucible_bench.compare import compare

        run = _harbor_run(tmp_path, [
            _trial("a", reward=1.0), _trial("b", reward=1.0),
            _trial("c", reward=0.0), _trial("d", reward=0.0),
        ])
        imported = import_tb(run, None)

        def as_result(model, passed, total):
            return {
                "model": model,
                "timestamp": "2026-08-22T11:30:00",
                "results": {imported.task: {
                    "score": passed / total, "passed": passed, "total": total,
                }},
            }

        comp = compare(as_result("baseline", 3, 4), as_result("reap-20pct", 2, 4))
        assert comp["tasks"]["terminal_bench"]["delta"] == pytest.approx(-0.25)


class TestCliDispatch:
    """A subcommand flag must not be able to shadow the dispatch target.

    `--command` on `import` originally used argparse's default dest, which is
    the same name the subparsers use. Passing it worked; omitting it set the
    dispatch target to None and the CLI printed help and exited 1 instead of
    importing — for a run that had already cost hours of container time.
    """

    def test_import_dispatches_without_optional_flags(self):
        from crucible_bench.cli import build_parser

        args = build_parser().parse_args(
            ["import", "--tool", "swebench", "--from", "x", "--model", "y"]
        )
        assert args.subcommand == "import"
        assert callable(args.run)

    def test_import_dispatches_with_command_flag(self):
        from crucible_bench.cli import build_parser

        args = build_parser().parse_args(
            ["import", "--tool", "swebench", "--from", "x", "--model", "y",
             "--command", "harbor run -d terminal-bench@2.0"]
        )
        assert args.subcommand == "import"
        assert args.command_line == "harbor run -d terminal-bench@2.0"
        assert callable(args.run)

    def test_no_subcommand_flag_shadows_the_dispatch_target(self):
        # Structural: any future flag named --command anywhere would reintroduce
        # this, so assert the dispatch name is not one a flag can claim.
        from crucible_bench.cli import COMMANDS, build_parser

        parser = build_parser()
        for command in COMMANDS:
            sub = argparse_subparser(parser, command.NAME)
            dests = {a.dest for a in sub._actions}
            assert "subcommand" not in dests, (
                f"{command.NAME} declares a flag with dest 'subcommand', which "
                f"would shadow CLI dispatch"
            )


def argparse_subparser(parser, name):
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices and name in action.choices:
            return action.choices[name]
    raise KeyError(name)
