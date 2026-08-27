# crucible-bench

Benchmark a served model over an OpenAI-compatible API — code generation, tool
calling, instruction following, multiple choice.

Two dependencies, both for pulling datasets. Everything that talks to a model is
stdlib `urllib`, so this installs on anything that can reach an endpoint: a
laptop, a CI runner, a box with no GPU and no compiler.

```bash
uv pip install crucible-bench

crucible-bench run --url http://localhost:8091 --model baseline --suite regression -o results/
crucible-bench run --url http://localhost:8091 --model reap-40pct --suite regression -o results/
crucible-bench compare results/baseline_*.json results/reap-40pct_*.json --markdown
```

It is the measurement half of [crucible](https://github.com/peonist-ai/crucible), a
MoE compression toolkit, but it has no dependency on it and does not care how
the model on the other end of the socket was made.

## Benchmarks

| Task | What it is |
|---|---|
| `humaneval`, `humaneval_plus` | 164 problems, pass@1, executed against the dataset's tests |
| `mbpp`, `mbpp_plus` | 500 / 378 problems, pass@1 |
| `bigcodebench` | 1140 real-library problems, pass@1 |
| `bfcl_simple` | 400 single-function tool calls, BFCL AST match |
| `bfcl_multi_turn[_*]` | 200 entries each × 5 categories, stateful tool use over 4–6 turns, state-graded (needs `[bfcl]`) |
| `ifeval` | 541 prompts / 834 verifiable instructions, strict prompt- and instruction-level |
| `mmlu_pro` | 10-option MC, 5-shot CoT |
| `gpqa_diamond` | 198 graduate-level science MC, 0-shot CoT |

Suites: `quick`, `coding`, `coding_plus`, `regression`, `agentic`, `instruct`, `full`.

`bfcl_multi_turn` is the one that measures error accumulation. Single-call tool
use barely moves under compression; multi-step application accuracy drops
several times as much, and nothing else here can see that. Each entry runs 4–6
turns against seeded stateful APIs, and the result records **which turn**
diverged — an entry that dies at turn 1 and one that holds until turn 5 are
different diagnoses that a pass rate cannot tell apart.

## Honesty rules

A check this harness cannot actually perform raises. It never scores as
"passed", and it never guesses on the model's behalf:

- An unknown IFEval instruction id, a missing `langdetect`/`nltk`, or malformed
  BFCL ground truth is an error, preflighted before the first request rather
  than discovered an hour in.
- Unparseable multiple-choice output is scored wrong *and* recorded
  (`extraction_tier`), and the run summary reports how many (`unparsed`)
  alongside how many hit the token cap (`truncated`).
- IFEval's 25 checkers are ported from the
  [reference implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
  one at a time, not reconstructed from memory.
- `livecodebench` raises. It has no scorer, and a task with no scorer does not
  get to report a number.
- `bfcl_multi_turn` scores are **ours, not BFCL's**. We drive generation
  ourselves to capture per-turn divergence, so reproduce through
  `bfcl evaluate` before putting a number next to a leaderboard entry.

## Importing external benchmarks

Some benchmarks we deliberately don't run. Terminal-Bench needs Harbor to build
and drive containers; SWE-bench needs its own harness to apply patches and run
repo test suites. Reimplementing either would mean claiming to measure something
we actually delegated.

So run them with their own harness and import the result:

```bash
# Terminal-Bench 2.0, via Harbor
harbor run -d terminal-bench@2.0 -a terminus-2 -m hosted_vllm/reap-20pct

crucible-bench import --tool terminal-bench --from runs/<run-id>/ \
  --model reap-20pct --dataset 'terminal-bench@2.0' --agent 'terminus-2 0.4.1' \
  --endpoint http://your-server:8000/v1 -o results/

# SWE-bench, after mini-swe-agent + the official evaluation harness
crucible-bench import --tool swebench --from reap-20pct.run1.json \
  --model reap-20pct --task swebench_verified_mini \
  --dataset 'MariusHobbhahn/swe-bench-verified-mini' \
  --agent 'mini-swe-agent 1.2.0' -o results/
```

The output is the same shape `run` writes, so `compare` diffs an imported
Terminal-Bench score against a native HumanEval+ one without caring which is
which. Two differences that do matter:

- `"imported": true` and a `provenance` block naming every variable the external
  tool controlled and we did not — harness, agent scaffold, dataset version,
  the exact command line. An imported score without that is a number you cannot
  check.
- `incomplete` counts trials that produced no verdict: a container that failed
  to build, an agent that crashed, a patch that never applied. They stay inside
  the denominator and score as failures — that is what the benchmarks do — but
  they are counted separately so a run wrecked by infrastructure can't be read
  as a model that got worse.

Where the tool reports its own rate (Harbor's `pass_at_k`), we record it beside
ours rather than replacing it, so a disagreement is visible instead of silently
resolved in our favour. And where a trial reports several rewards with no way to
tell which means "solved", the import fails rather than picking one.

See [docs/external-benchmarks.md](../../docs/external-benchmarks.md) for the full runbooks.

## Extras

```bash
uv pip install 'crucible-bench[ifeval]'   # langdetect + nltk, required to score IFEval
python -m nltk.downloader punkt punkt_tab

uv pip install 'crucible-bench[bfcl]'     # simulated APIs, required to score bfcl_multi_turn
uv pip install 'crucible-bench[local]'    # lm-eval, for `eval-local` over a checkpoint
```

`eval-local` is the odd one out: it loads weights instead of calling an
endpoint, which is why the torch it drags in is opt-in.

## Sandboxing

Coding benchmarks execute code a language model wrote. `--sandbox auto` (the
default) containerizes with podman or docker when either is present and warns
loudly when it falls back to running locally. `--sandbox docker` refuses to run
without isolation — use that against any endpoint you do not control.

## License

Apache-2.0. See [LICENSE](LICENSE).
