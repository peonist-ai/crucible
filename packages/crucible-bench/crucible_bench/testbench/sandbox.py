"""Isolation for model-generated code.

Coding benchmarks work by running code a language model wrote. That is
arbitrary code execution by construction, and the model is only as
trustworthy as the endpoint you pointed at. Running it in a container with no
network is the difference between "evaluated a model" and "ran a stranger's
program as yourself".

Configured once by the caller rather than threaded through nine benchmark
functions; see set_sandbox().
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SANDBOX_MODES = ("auto", "docker", "podman", "none")
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"

_sandbox_runtime: str | None = None       # resolved runtime, or None for local
_sandbox_image: str = DEFAULT_SANDBOX_IMAGE
_sandbox_warned: bool = False


def set_sandbox(mode: str = "auto", image: str = DEFAULT_SANDBOX_IMAGE) -> str | None:
    """Choose how generated code is executed. Returns the runtime, or None.

    "auto" prefers podman, then docker, and falls back to local execution with
    a warning when neither is installed. "docker"/"podman" require that runtime
    and raise if it is missing — use those in CI or against untrusted models.
    "none" runs locally, no warning, which is only reasonable for a model you
    compressed yourself.
    """
    global _sandbox_runtime, _sandbox_image, _sandbox_warned

    if mode not in SANDBOX_MODES:
        raise ValueError(f"sandbox mode must be one of {SANDBOX_MODES}, got {mode!r}")

    _sandbox_image = image
    _sandbox_warned = False

    if mode == "none":
        _sandbox_runtime = None
        return None

    candidates = [mode] if mode in ("docker", "podman") else ["podman", "docker"]
    for runtime in candidates:
        if shutil.which(runtime):
            _sandbox_runtime = runtime
            return runtime

    if mode != "auto":
        raise RuntimeError(
            f"--sandbox {mode} requested but {mode} is not on PATH. Install it, "
            f"or pass --sandbox none to accept running generated code locally."
        )

    _sandbox_runtime = None
    return None


def _sandbox_argv(
    runtime: str,
    host_dir: str,
    filename: str,
    *,
    writable: bool = False,
    args: list[str] | None = None,
) -> list[str]:
    """Container invocation: no network, read-only root, capped memory/pids.

    `writable` mounts /work rw. Benchmarks keep it read-only — generated code has
    no business editing the problem. Agent probes need it: the whole task is the
    model editing a workspace, and its tests write `__pycache__` alongside them.
    The isolation that matters either way is the network, memory and pid caps,
    none of which this loosens.
    """
    return [
        runtime, "run", "--rm",
        "--network", "none",          # no exfiltration, no callbacks
        "--read-only",                # rootfs immutable
        "--tmpfs", "/tmp:rw,size=64m",
        "--memory", "512m",
        "--pids-limit", "128",        # fork bombs terminate themselves
        "--workdir", "/work",
        # :Z relabels the bind mount for SELinux. Without it, on any enforcing
        # system (Fedora, RHEL) the container gets EACCES on every file in /work
        # and EVERY problem fails — a 0% that looks exactly like the model
        # collapsing. Measured on the Strix Halo 2026-08-25. Runtimes without
        # SELinux accept the suffix and ignore it, so it is safe everywhere.
        "-v", f"{host_dir}:/work:{'rw' if writable else 'ro'},Z",
        _sandbox_image,
        "python", f"/work/{filename}",
        *(args or []),
    ]


class SandboxMissingDependency(RuntimeError):
    """The sandbox image cannot import what a benchmark's tests need."""


def require_modules(modules: list[str]) -> None:
    """Refuse to run a code benchmark the sandbox cannot possibly pass.

    EvalPlus's *tests* import numpy — 163 of 164 HumanEval+ problems and all
    378 MBPP+ problems. Under the default `python:3.12-slim` every one of them
    dies on `import numpy` inside the test block, so the task scores 0% no
    matter what the model wrote. Measured 2026-08-22: a correct HumanEval/0
    solution failed with ModuleNotFoundError raised by the harness, not the
    solution.

    That is a scored-zero caused by our configuration, which is the worst kind
    of wrong number: it looks exactly like catastrophic model failure. So we
    probe the image once, up front, and refuse rather than produce it.

    Local execution is not probed — if the module were missing the benchmark
    would fail the same way, but the fix there is the caller's environment, and
    raising on it would break the no-container path for no benefit.
    """
    # Nothing to check means nothing to probe. Without this the function spawns
    # a container to run `print('ok')` and can fail on a missing image — a task
    # that declares no test-time imports has no reason to depend on the sandbox
    # being buildable at all.
    if not modules or not _sandbox_runtime:
        return

    # The probe runs a real FILE through a real BIND MOUNT, not `python -c`.
    # An earlier version imported the modules inline and passed happily on a
    # host where every mounted file was unreadable (SELinux enforcing, no :Z),
    # so the preflight said "fine" and all 164 problems then scored 0. Probe the
    # path the benchmark actually uses, or the probe is theatre.
    probe = "".join(f"import {m}\n" for m in modules) + "print('ok')\n"
    with tempfile.TemporaryDirectory() as workdir:
        Path(workdir, "probe.py").write_text(probe)
        argv = _sandbox_argv(_sandbox_runtime, workdir, "probe.py")
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except Exception as e:
            raise SandboxMissingDependency(
                f"could not probe sandbox image {_sandbox_image!r}: {e}"
            ) from e
        if result.returncode == 0 and result.stdout.strip().endswith("ok"):
            return
        if "Permission denied" in result.stderr or "can't open file" in result.stderr:
            raise SandboxMissingDependency(
                f"sandbox {_sandbox_runtime!r} cannot READ the mounted work "
                f"directory, so every problem would fail on the file itself and "
                f"the task would score 0%.\n"
                f"On SELinux hosts a bind mount needs a relabel suffix (:Z); this "
                f"build passes one, so a failure here means something else is "
                f"blocking the mount.\n"
                f"Probe stderr: {result.stderr.strip()[:300]}"
            )
    if modules:
        raise SandboxMissingDependency(
            f"sandbox image {_sandbox_image!r} cannot import {modules}, which "
            f"this benchmark's own tests require — every problem would fail on "
            f"the import and the task would score 0%.\n"
            f"For the usual case (numpy, what EvalPlus needs) the repo ships one:\n"
            f"    docker build -t crucible-bench-sandbox "
            f"-f containers/bench-sandbox.Dockerfile .\n"
            f"Otherwise build an image that has them, e.g.:\n"
            f"    docker build -t crucible-bench-sandbox - <<'EOF'\n"
            f"    FROM python:3.12-slim\n"
            f"    RUN pip install --no-cache-dir {' '.join(modules)}\n"
            f"    EOF\n"
            f"then pass --sandbox-image crucible-bench-sandbox.\n"
            f"Probe stderr: {result.stderr.strip()[:300]}"
        )


def execute_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute model-generated Python and return (passed, error_message).

    Isolation depends on set_sandbox(); the default falls back to local
    execution when no container runtime exists, warning once.
    """
    global _sandbox_warned

    with tempfile.TemporaryDirectory() as workdir:
        filename = "snippet.py"
        path = Path(workdir) / filename
        path.write_text(code)

        if _sandbox_runtime:
            argv = _sandbox_argv(_sandbox_runtime, workdir, filename)
            # Container startup is ~0.3s; give it room beyond the code's budget.
            wall_timeout = timeout + 30
        else:
            if not _sandbox_warned:
                print(
                    "  WARNING: running model-generated code directly on this "
                    "machine with no isolation. Install podman or docker, or "
                    "pass --sandbox docker, to contain it.",
                    file=sys.stderr,
                )
                _sandbox_warned = True
            argv = [sys.executable, filename]
            wall_timeout = timeout

        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=wall_timeout,
                cwd=workdir,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)


def run_file(
    host_dir: str,
    filename: str,
    args: list[str] | None = None,
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Run a file that already exists in `host_dir`, returning (exit_code, stdout, stderr).

    The agent-probe counterpart to execute_code(). The difference is ownership of
    the workspace: execute_code() writes one throwaway snippet and reads a
    pass/fail, while here the model has spent a whole session editing a directory
    and we run something out of it.

    That distinction is why this exists rather than the harness calling
    subprocess directly. An agent loop that hands the model `write_file` and then
    executes what it wrote is arbitrary code execution with extra steps — the
    path check that keeps writes inside the workspace says nothing about what the
    code does once it runs. Same isolation, same fallback, same one-time warning
    as every other benchmark in this package.
    """
    global _sandbox_warned

    if _sandbox_runtime:
        argv = _sandbox_argv(
            _sandbox_runtime, host_dir, filename, writable=True, args=args
        )
        wall_timeout = timeout + 30
    else:
        if not _sandbox_warned:
            print(
                "  WARNING: running model-generated code directly on this "
                "machine with no isolation. Install podman or docker, or "
                "pass --sandbox docker, to contain it.",
                file=sys.stderr,
            )
            _sandbox_warned = True
        argv = [sys.executable, filename] + list(args or [])
        wall_timeout = timeout

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=wall_timeout, cwd=host_dir
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"TIMEOUT after {wall_timeout}s"
    except Exception as e:
        return 1, "", str(e)
