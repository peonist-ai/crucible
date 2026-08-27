"""Agent probes: the bundled tasks, and the isolation the harness must not lose.

These probes moved into this package from `scripts/` because measuring a served
model is this package's job. The move mattered for one concrete reason: the
harness ran the model's code with `subprocess.run([sys.executable, ...])`
directly on the host. Its path check kept *writes* inside the workspace and said
nothing about what the code did once running.
"""

import inspect
from pathlib import Path

import pytest

from crucible_bench.agent_probes import harness

TASKS = Path(harness.__file__).parent / "tasks"


class TestBundledTasks:
    """The tasks are package data, which is silently droppable at build time."""

    def test_tasks_are_present(self):
        names = {p.name for p in TASKS.iterdir() if p.is_dir()}
        assert names == {"bugfix", "feature_add", "recovery"}

    @pytest.mark.parametrize("task", ["bugfix", "feature_add", "recovery"])
    def test_each_task_is_runnable(self, task):
        # main() exits on either of these; a task missing one is a task that
        # can never run, and package-data misconfiguration looks exactly like it.
        assert (TASKS / task / "task.md").is_file()
        assert (TASKS / task / "sandbox").is_dir()

    @pytest.mark.parametrize("task", ["bugfix", "feature_add", "recovery"])
    def test_each_task_can_be_graded(self, task):
        # The harness records pass/fail by running test_*.py in the workspace.
        # Without one there is no signal, only a transcript.
        tests = list((TASKS / task / "sandbox").glob("test_*.py"))
        assert tests, f"{task} has no test_*.py, so it cannot be scored"


class TestModelCodeStaysContained:
    def test_harness_does_not_execute_code_itself(self):
        # The regression guard. If `subprocess` reappears here, someone has
        # re-added a path that runs model-written code on the host.
        source = Path(harness.__file__).read_text()
        assert "import subprocess" not in source
        assert "subprocess.run" not in source

    def test_both_execution_paths_go_through_the_sandbox(self):
        source = Path(harness.__file__).read_text()
        # One for the `run_python` tool, one for the final test pass.
        assert source.count("sandbox.run_file(") == 2

    def test_run_python_still_rejects_path_escapes(self, tmp_path):
        # Containment did not replace the path check — a probe that let the
        # model name /etc/passwd would mount-escape before isolation applied.
        with pytest.raises(ValueError, match="escapes sandbox"):
            harness.safe_path(str(tmp_path), "../../etc/passwd")

    def test_sandbox_mode_is_selectable_from_the_cli(self):
        # Against an endpoint you do not control, 'auto' silently falling back
        # to local execution is the wrong default to be stuck with.
        source = inspect.getsource(harness.main)
        assert "--sandbox" in source
        assert "set_sandbox" in source
