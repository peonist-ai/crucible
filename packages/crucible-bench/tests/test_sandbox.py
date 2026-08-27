"""Sandbox selection for model-generated code.

These tests cover the decision logic and the container invocation, not actual
container execution — CI has no guarantee of a runtime, and the point is that
the *choice* is never silently wrong.
"""

import pytest

from crucible_bench.testbench import sandbox


@pytest.fixture(autouse=True)
def restore_sandbox():
    yield
    sandbox.set_sandbox("none")


class TestSandboxSelection:
    def test_none_disables_isolation(self):
        assert sandbox.set_sandbox("none") is None

    def test_auto_falls_back_when_no_runtime(self, monkeypatch):
        monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)

        # Falling back is deliberate: requiring a container by default would
        # break benchmarking on hosts that intentionally have no docker.
        assert sandbox.set_sandbox("auto") is None

    def test_auto_prefers_podman(self, monkeypatch):
        monkeypatch.setattr(sandbox.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

        assert sandbox.set_sandbox("auto") == "podman"

    def test_auto_uses_docker_when_podman_absent(self, monkeypatch):
        monkeypatch.setattr(
            sandbox.shutil, "which",
            lambda cmd: "/usr/bin/docker" if cmd == "docker" else None,
        )

        assert sandbox.set_sandbox("auto") == "docker"

    def test_explicit_runtime_raises_when_missing(self, monkeypatch):
        # An explicit --sandbox docker is a security requirement, so silently
        # running unisolated would defeat the request.
        monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)

        with pytest.raises(RuntimeError, match="not on PATH"):
            sandbox.set_sandbox("docker")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="sandbox mode"):
            sandbox.set_sandbox("chroot")


class TestContainerInvocation:
    def _argv(self):
        return sandbox._sandbox_argv("podman", "/tmp/work", "snippet.py")

    def test_network_is_disabled(self):
        argv = self._argv()
        assert "--network" in argv
        assert argv[argv.index("--network") + 1] == "none"

    def test_filesystem_is_read_only(self):
        argv = self._argv()
        assert "--read-only" in argv
        assert "-v" in argv
        mount = argv[argv.index("-v") + 1]
        assert ":ro" in mount

    def test_bind_mount_is_selinux_relabelled(self):
        """Without :Z, an enforcing host gives EACCES on every file in /work.

        Measured on the Strix Halo (Fedora, SELinux Enforcing) 2026-08-25: the
        container could not open the mounted snippet at all, so all 164
        HumanEval+ problems failed and the task scored 0% — indistinguishable
        from total model collapse. Runtimes without SELinux ignore the suffix.
        """
        mount = self._argv()[self._argv().index("-v") + 1]
        assert mount.endswith(":ro,Z"), mount

    def test_resources_are_capped(self):
        argv = self._argv()
        assert "--memory" in argv
        assert "--pids-limit" in argv

    def test_container_is_removed(self):
        assert "--rm" in self._argv()

    def test_image_is_configurable(self):
        sandbox.set_sandbox("none", image="my/bigcodebench:latest")
        assert "my/bigcodebench:latest" in self._argv()


class TestRequireModules:
    """A sandbox that cannot import the tests' dependencies must refuse.

    Found on a real run 2026-08-22: under the default `python:3.12-slim`, a
    correct HumanEval/0 solution failed with ModuleNotFoundError raised by the
    *test* block, not the solution. 163/164 HumanEval+ and 378/378 MBPP+ tests
    import numpy, so both tasks scored 0% for a configuration reason that looks
    identical to catastrophic model failure.
    """

    def test_local_execution_is_not_probed(self, monkeypatch):
        monkeypatch.setattr(sandbox, "_sandbox_runtime", None)
        sandbox.require_modules(["definitely_not_installed_xyz"])  # no raise

    def test_empty_module_list_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")
        sandbox.require_modules([])  # no raise, no subprocess

    def test_missing_module_raises_with_a_fix(self, monkeypatch):
        import subprocess

        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")
        monkeypatch.setattr(sandbox, "_sandbox_image", "python:3.12-slim")

        def fake_run(argv, **kw):
            return subprocess.CompletedProcess(
                argv, 1, "", "ModuleNotFoundError: No module named 'numpy'"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(sandbox.SandboxMissingDependency) as e:
            sandbox.require_modules(["numpy"])
        # The message has to carry the fix, not just the complaint.
        assert "--sandbox-image" in str(e.value)
        assert "score 0%" in str(e.value)

    def test_present_module_passes(self, monkeypatch):
        import subprocess

        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")

        def fake_run(argv, **kw):
            return subprocess.CompletedProcess(argv, 0, "ok\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        sandbox.require_modules(["numpy"])  # no raise

    def test_probe_runs_with_no_network(self, monkeypatch):
        import subprocess

        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")
        seen = {}

        def fake_run(argv, **kw):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, "ok\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        sandbox.require_modules(["numpy", "pandas"])
        assert "--network" in seen["argv"] and "none" in seen["argv"]

    def test_probe_exercises_a_real_bind_mount(self, monkeypatch):
        """`python -c` would pass on a host where every mounted file is unreadable.

        That is exactly what happened: the preflight said fine, then all 164
        problems scored 0 on EACCES. The probe must use the path the benchmark
        actually uses.
        """
        import subprocess

        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")
        seen = {}

        def fake_run(argv, **kw):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, "ok\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        sandbox.require_modules(["numpy"])
        assert "-v" in seen["argv"], "probe never mounted anything"
        assert "-c" not in seen["argv"], "probe still uses inline python -c"

    def test_unreadable_mount_raises_rather_than_scoring_zero(self, monkeypatch):
        import subprocess

        monkeypatch.setattr(sandbox, "_sandbox_runtime", "docker")

        def fake_run(argv, **kw):
            return subprocess.CompletedProcess(
                argv, 1, "", "python: can't open file '/work/probe.py': "
                             "[Errno 13] Permission denied")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(sandbox.SandboxMissingDependency, match="cannot READ"):
            sandbox.require_modules(["numpy"])


class TestRunFile:
    """`run_file` is the agent probes' execution path.

    It exists because a harness that hands the model `write_file` and then runs
    what it wrote is arbitrary code execution — the harness used to call
    subprocess directly, on the host, with no isolation at all.
    """

    def test_workspace_is_writable_unlike_benchmarks(self):
        # Benchmarks mount ro: generated code has no business editing the
        # problem. Agent probes must mount rw, because editing IS the task —
        # and their tests write __pycache__ next to the files they import.
        agent = sandbox._sandbox_argv("docker", "/w", "t.py", writable=True)
        bench = sandbox._sandbox_argv("docker", "/w", "t.py")
        assert "/w:/work:rw,Z" in agent
        assert "/w:/work:ro,Z" in bench

    def test_writable_does_not_loosen_the_real_containment(self):
        argv = sandbox._sandbox_argv("docker", "/w", "t.py", writable=True)
        # The mount is the only thing that changes. Network, memory and pid
        # caps are what actually contain the code, and they still apply.
        assert "none" in argv and "--network" in argv
        assert "512m" in argv
        assert "128" in argv

    def test_script_arguments_are_forwarded(self):
        argv = sandbox._sandbox_argv("docker", "/w", "t.py", args=["--flag", "v"])
        assert argv[-3:] == ["/work/t.py", "--flag", "v"]

    def test_runs_locally_when_isolation_is_disabled(self, tmp_path):
        sandbox.set_sandbox("none")
        (tmp_path / "ok.py").write_text("import sys; sys.exit(7)")
        code, _, _ = sandbox.run_file(str(tmp_path), "ok.py")
        assert code == 7

    def test_stdout_and_stderr_are_returned_separately(self, tmp_path):
        sandbox.set_sandbox("none")
        (tmp_path / "s.py").write_text(
            "import sys; print('OUT'); print('ERR', file=sys.stderr)"
        )
        code, out, err = sandbox.run_file(str(tmp_path), "s.py")
        assert code == 0 and "OUT" in out and "ERR" in err

    def test_timeout_is_reported_not_raised(self, tmp_path):
        # A model that writes an infinite loop must not hang the probe run.
        sandbox.set_sandbox("none")
        (tmp_path / "hang.py").write_text("while True: pass")
        code, _, err = sandbox.run_file(str(tmp_path), "hang.py", timeout=1)
        assert code == 124 and "TIMEOUT" in err
