# Agent probes

Small multi-turn agentic tasks for evaluating a chat-completions endpoint.
Each task is self-contained: a sandbox of starter files and a `task.md` prompt.
The harness gives the model `read_file`, `write_file`, `list_files`, and
`run_python` tools, runs the loop until the model stops calling tools, then
records pass/fail of any `test_*.py` files in the sandbox.

## Run

```bash
# once — two of the tasks' tests need pytest
docker build -t crucible-bench-agent-probe -f containers/agent-probe-sandbox.Dockerfile .

python -m crucible_bench.agent_probes.harness \
  --task packages/crucible-bench/crucible_bench/agent_probes/tasks/bugfix \
  --url http://localhost:8093/v1 \
  --sandbox docker \
  --transcript-out runs/
```

The `--url` should be the `/v1` base of an OpenAI-compatible server (vLLM,
llama-server, etc.). The task's `sandbox/` is copied to `/tmp/agent_probe_<task>`
so the repo stays clean.

## Isolation

The model gets `run_python`, and its own tests are executed at the end. Both go
through `crucible_bench.testbench.sandbox`, the same containment every other
benchmark in this package uses: `--network none`, a 512 MB memory cap, and a
128-process limit. The workspace is mounted read-write here — unlike the
benchmark sandbox — because editing it is the task.

`--sandbox auto` (the default) falls back to running that code directly on your
machine, with a warning, when no container runtime is present. Pass
`--sandbox docker` or `--sandbox podman` against any endpoint you do not
control; those fail rather than fall back.

## Tasks

- **`bugfix`** — single-file off-by-operator bug (`+` instead of `*`). Tests
  basic agentic loop: read → diagnose → fix → verify.
- **`feature_add`** — add a `power` operation across `ops.py`, `main.py`, and
  `test_ops.py`. Tests multi-file consistency and style matching.
- **`recovery`** — deep-merge bug where the obvious first-attempt fix (naive
  recursion) is likely to break a different test case. Tests whether the
  model debugs its own incorrect output.

## Adding a task

1. `mkdir tasks/<name>/sandbox`
2. Drop starter files in `sandbox/` (the agent will edit copies of these)
3. Write `tasks/<name>/task.md` with the user-facing prompt
4. Make sure any test files use `test_*.py` naming so the harness picks them
   up for the final pass/fail check
