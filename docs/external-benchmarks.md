# Agentic benchmark runbooks

Terminal-Bench and SWE-bench run under their own harnesses, not under
crucible-bench. Nothing here is imported by either package — these are
invocation notes, and the result files they produce get pulled in with
`crucible-bench import`.

Why they live outside: crucible-bench implements a benchmark only when it owns
the whole loop — endpoint in, grading in-process, results out. It cannot grade
either of these, and a task registered as first-class while something else does
the grading is how `livecodebench` ended up reporting `"def " in response` as
pass@1.

## Terminal-Bench 2.0

Harbor is the official harness. It installs as an isolated tool — nothing enters
either package's dependency tree.

```bash
uv tool install harbor
```

**Do the oracle smoke test first.** The `oracle` agent runs each task's reference
solution, so it validates dataset download, container startup and — the part that
actually bites — whether these images build on your architecture, at zero
inference cost:

```bash
harbor run -d terminal-bench/terminal-bench-2 -a oracle -l 5
```

Then the real run. Terminus-2 drives a tmux session and parses commands from
text (`parser_name` is `json` or `xml`), so it does not need native tool calling
— useful, because it isolates agentic reasoning from tool-call formatting:

```bash
harbor run -d terminal-bench@2.0 -a terminus-2 \
  -m hosted_vllm/qwen36-reap-20pct
# api_base goes through the agent kwargs, e.g. "api_base": "http://your-server:8000/v1"
```

Output lands in a run directory containing `results.json` (a `JobResult`),
`run_metadata.json`, `run.log`, and per-trial directories with pane snapshots
and asciinema recordings.

```bash
crucible-bench import --tool terminal-bench --from <run-dir>/ \
  --model qwen36-reap-20pct --dataset 'terminal-bench@2.0' \
  --agent 'terminus-2 <version>' --endpoint http://your-server:8000/v1 \
  --command 'harbor run -d terminal-bench@2.0 -a terminus-2 -m ...' \
  -o results/
```

**Read the score knowing this:** frontier agents resolve under 65% and small
models land around 15%. At ~89 tasks that is a floor-effect benchmark for a
compressed 35B — a marginal-rate delta will be inside the noise. It earns its
place as a model-card number and a milestone gate, not as a per-variant
regression signal. For actual signal, use the per-task detail the importer
preserves and compare fail-sets pairwise.

## SWE-bench

Two steps with two different failure modes: the agent produces patches, then a
separate harness grades them.

```bash
uv tool install mini-swe-agent
```

mini-swe-agent is ~100 lines, bash-only, and its default path parses commands
out of plain text rather than using the tool-calling interface. That is the
reason to run it alongside BFCL: together they separate "lost the ability to
plan and debug over 30 turns" from "lost the ability to emit a well-formed tool
call". **Confirm which agent class you instantiated** — the README and the
local-models docs disagree, and the distinction is the whole point.

```bash
mini-extra swebench \
  --subset MariusHobbhahn/swe-bench-verified-mini --split test \
  -m hosted_vllm/qwen36-reap-20pct -w 4 -o runs/reap-20pct/
```

Config for a local endpoint:

```yaml
model:
  model_name: "hosted_vllm/qwen36-reap-20pct"
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "http://your-server:8000/v1"
```

plus `MSWEA_COST_TRACKING=ignore_errors`, or a `LITELLM_MODEL_REGISTRY_PATH`
JSON with zero costs.

Then grade:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name MariusHobbhahn/swe-bench-verified-mini \
  --predictions_path runs/reap-20pct/preds.json \
  --run_id reap-20pct-1

crucible-bench import --tool swebench --from qwen36-reap-20pct.reap-20pct-1.json \
  --model qwen36-reap-20pct --task swebench_verified_mini \
  --dataset 'MariusHobbhahn/swe-bench-verified-mini' \
  --agent 'mini-swe-agent <version>' -o results/
```

### Architecture, which is the real blocker

SWE-bench's prebuilt Docker images are **x86_64 only**. arm64 is experimental:
images must be built locally with `--namespace ""`, and there are known failures
(the Chrome apt path is invalid on ARM, `mvnd` ships no Linux ARM64 binary).
Epoch AI publishes ~1,819 of 2,294 arm64 images best-effort and untested.

So the M4 Mac is the wrong host for this one, unlike Terminal-Bench:

- **Podman on the Halo** — native x86_64, no emulation, and `sandbox.py` already
  prefers podman. Contends with vLLM for the 128GB.
- **Mac under emulation** — works, slow, flaky. Flaky test results are
  indistinguishable from compression damage, which is the one confound this
  whole exercise exists to avoid.
- **`sb-cli` cloud grading** — rollout stays local, only patches are graded
  remotely. Sidesteps architecture entirely, but ships your patches off-box.

`swe-bench-verified-mini` is 50 tasks needing **5GB instead of 130GB**, selected
by linear programming plus k-means to match the full 500's difficulty and
pass-rate distribution. That is what makes any of this affordable.

At n=50, the standard error near p=0.5 is about 7 points — a 5-point delta is
not a result. Use the paired fail-set comparison, not marginal rates.
