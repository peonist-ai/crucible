<p align="center">
  <img src="docs/crucible-banner.jpg"
       alt="Crucible — Mixture-of-Expert compression for consumer hardware"
       width="640">
</p>

# Crucible

MoE expert compression toolkit — prune, merge, and forge specialist models for local inference.

Takes large Mixture-of-Experts language models and compresses them into compact specialists
that run on consumer hardware. Implements [REAP](https://arxiv.org/abs/2510.13999) expert
pruning with several scoring strategies, plus a self-contained evaluation harness so every
compression run can be measured against its baseline.

## Why

A Mixture-of-Experts model carries every expert's weights but activates a handful per token —
8 of 128, 8 of 256, 2 of 8, depending on the architecture. For any given specialist role, most
of that resident weight is dead. Crucible scores each expert by how much work it actually does
on calibration data representative of the target role, removes the ones that don't earn their
place, and leaves a model that quantizes and serves normally.

This is not tied to one model family. Any MoE architecture works once it has a registry entry,
and adding one is a single `ModelAttrs` declaration — no CLI change, no method change.

## Install

This repo is a uv workspace holding two packages, because compression and measurement are
separate concerns with separate dependency trees:

| Package | Does | Needs |
|---|---|---|
| `crucible` | prunes, merges, quantizes — produces weights | torch, transformers |
| `crucible-bench` | drives an OpenAI-compatible endpoint — produces scores | `datasets`, `huggingface-hub` |

They share no imports. The only thing crossing the boundary is files: crucible writes a model and
its `compression_metadata.json`, crucible-bench writes result JSON.

```bash
# both, for development
uv sync
uv run crucible --help
uv run crucible-bench --help

# just the benchmark harness — no torch, installs on a laptop that only
# ever talks to a remote server
uv pip install crucible-bench
```

Requires Python 3.12+. `crucible` needs `transformers>=5.5` (Gemma 4 and Qwen 3.5/3.6 MoE
modeling code does not exist in 4.x).

## Use

```bash
# What does this model's MoE look like?
uv run crucible inspect Qwen/Qwen3.6-35B-A3B

# Score experts once, reuse the scores across every compression ratio
uv run crucible observe Qwen/Qwen3.6-35B-A3B --samples 512 -o results/observation.json

# Prune 40% of experts using those scores
uv run crucible compress Qwen/Qwen3.6-35B-A3B \
  --method reap --ratio 0.40 \
  --scores-file results/observation.json \
  -o outputs/reap-40pct

# Quantize the pruned model to compressed-tensors W4A16 (needs llm-compressor)
uv run crucible quantize outputs/reap-40pct/<model> -o outputs/reap-40pct-w4a16

# Benchmark a served model (any OpenAI-compatible endpoint)
uv run crucible-bench run --url http://localhost:8091 --model reap-40pct \
  --suite regression -o results/

# Diff two runs, with recovery percentages
uv run crucible-bench compare results/baseline.json results/reap-40pct.json
uv run crucible-bench compare results/baseline.json results/reap-40pct.json --markdown
```

Pruning and quantization are separate passes on purpose — save between them, so a failed
quantization never costs the prune. The quantized output is compressed-tensors, which vLLM serves
natively; on ROCm/gfx1151 it is the only 4-bit layout that reaches the RDNA-tuned kernels.

Observation is the expensive step and it is ratio-independent — run it once, then sweep ratios
against the saved scores.

## Methods

**REAP** (`--method reap`) — score each expert by router gate-value × activation norm, averaged
over the tokens actually routed to it, then drop the lowest scorers per layer. This is the
validated path and the default. Scoring variants via `--scoring`:

| strategy | what it changes |
|---|---|
| `reap` | the paper's metric, unmodified |
| `pathfinder` | scores experts by the cross-layer paths they sit on, and plans a non-uniform per-layer budget |
| `task-aware` | boosts experts specific to a target task vs. a general corpus |
| `adaptive` | allocates the pruning budget across layers by relative cost |

`--routing-aware` composes with any of them: it protects experts whose tokens have nowhere
else to go, applied after scoring at the target ratio.

**REAM** (`--method ream`) — merge-then-prune. **Experimental and currently broken on Gemma 4**:
the merge assumes a linearity that SiLU-gated experts don't have. The implementation is kept
because the approach is sound for architectures that satisfy its assumptions, but it is not a
supported path today.

Adding a method or a scoring strategy is one entry in `crucible/methods/registry.py`. The entry
declares its own CLI flags, so `--method`, `--scoring` and their help text follow from the
registry — no edit to the CLI.

## Evaluation

`crucible-bench` is a self-contained harness — no lm-eval dependency — that drives any
OpenAI-compatible endpoint (vLLM, llama-server, …):

- **Coding**: HumanEval, HumanEval+, MBPP+, BigCodeBench-Complete
- **Tool use**: BFCL (simple), plus multi-turn agentic probes under
  `crucible_bench.agent_probes` — sandboxed tool-calling tasks with real pass/fail
- **Instruction following**: IFEval
- **Knowledge**: MMLU-Pro and GPQA-Diamond, as a sanity check rather than a target

Suites are preset (`--suite quick|coding|coding_plus|regression|instruct|full`). Results are
written as structured JSON, one file per run, so `compare` can compute deltas rather than
eyeballing them.

Anything the harness cannot actually verify raises rather than scoring as a pass — see
[crucible-bench's honesty rules](packages/crucible-bench/README.md#honesty-rules). Benchmarks it
does not own the grading loop for (Terminal-Bench, SWE-bench) stay outside it: run them with their
own harnesses and import the results, rather than pretending crucible-bench controlled variables
it did not.

The [evaluation protocol](CLAUDE.md#evaluation-protocol) — always benchmark the uncompressed
baseline first, hold hardware and sampling fixed, record everything — is not optional. The
calibration mix has a larger effect on specialist quality than the compression ratio does.

## Security

Two things here execute untrusted input by design. Both are opt-in or contained,
but you should know they exist before pointing this at anything.

**Coding benchmarks run code the model wrote.** HumanEval, MBPP+ and BigCodeBench
work by executing generated Python and checking whether the tests pass — that is
arbitrary code execution, and the code is only as trustworthy as the endpoint you
benchmarked. `crucible-bench` therefore sandboxes it:

```bash
# default: containerize if podman/docker is present, else warn and run locally
crucible-bench run --url http://localhost:8091 --model mine --suite regression

# require isolation — fails rather than falling back
crucible-bench run --sandbox docker --url http://untrusted:8000 --model theirs ...
```

The container runs with `--network none`, a read-only rootfs, a 512 MB memory
cap and a 128-process limit. `--sandbox none` disables it, which is only
reasonable for a model you compressed yourself. Note that BigCodeBench imports
real libraries, so it needs `--sandbox-image` pointing at an image that carries
pandas/numpy/flask rather than the default `python:3.12-slim`.

EvalPlus needs one too — its *tests* import numpy, so under a plain
`python:3.12-slim` every HumanEval+/MBPP+ problem scores 0% regardless of what the
model wrote. The harness refuses to start rather than report that. Build the
image this repo ships:

```bash
docker build -t crucible-bench-sandbox -f containers/bench-sandbox.Dockerfile .
crucible-bench run --sandbox docker --sandbox-image crucible-bench-sandbox ...
```

**`--trust-remote-code` is off.** Passing it executes custom modeling code
shipped inside a checkpoint. Every model in the registry loads without it; only
enable it for sources you trust.

To report a security issue, see [SECURITY.md](SECURITY.md) — please use a
private advisory rather than a public issue.

**Dependencies are kept deliberately small** — four direct (`torch`,
`transformers`, `datasets`, `numpy`), 56 packages in a default install. Every
direct dependency is a name someone can take over, so additions should be
argued for. The optional `eval` extra pulls `lm-eval` and roughly doubles the
tree; `crucible-bench` is self-contained and doesn't need it.

## Supported models

| Model | Experts | Active | Expert storage |
|---|---|---|---|
| Gemma 4 26B-A4B | 128 | 8 | tensor3d |
| Qwen 3.5 / 3.6 35B-A3B | 256 + 1 shared | 8 | tensor3d |
| Qwen3 30B-A3B | 128 | 8 | modulelist |
| Mixtral 8x7B | 8 | 2 | modulelist |

These are the architectures currently in the registry, not the limit of what the method
handles — the range above spans 8 to 256 experts, fused and separate `gate_up_proj`, and both
`modulelist` and stacked-3D expert storage. Adding another is one `ModelAttrs` entry in
`crucible/models/registry.py`, declaring where that architecture keeps its router, experts and
projections. Grouped routers (DeepSeek-V3-shaped) are supported via `n_group_key` /
`top_k_group_key`.

## Repo layout

```
packages/
  crucible/       compression — types, CLI, model registry, methods
  crucible-bench/ measurement — endpoint client, benchmarks, comparison
scripts/          pipeline tools: calibration data, role sensitivity, KLD corpus
containers/       Dockerfiles for the benchmark and agent-probe sandboxes
docs/             bit allocation, and why some benchmarks run outside this repo
```

This repo is the tool. Compressed models and the numbers they scored are published
separately — `results/` is where *your* runs land, and it is not tracked.

Each package carries its own `tests/`; `uv run pytest` from the root collects both.

## Status

Working and used in anger, but early: one validated model family, one validated method, APIs
subject to change. Contributions and replication attempts welcome.

## Contributing

Contributions and replication attempts are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, the package boundary CI enforces,
and the "no silent passes" rule the benchmark harness is built around.

**There is no CLA.** Apache-2.0 section 5 already licenses what you submit under
the project's own terms, so a pull request is all that is needed.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## References

- [REAP: Router-weighted Expert Activation Pruning](https://arxiv.org/abs/2510.13999) — Cerebras Research
- [REAM: Merging Improves Pruning of Experts](https://arxiv.org/abs/2604.04356) — Samsung SAIL Montreal
- [REAP reference implementation](https://github.com/CerebrasResearch/reap)
- [llm-compressor](https://github.com/vllm-project/llm-compressor) — ships an upstream REAP modifier as of 0.13.0

## License

Apache-2.0. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party attribution.

This covers the toolkit. Compressed weights inherit the license of the model they were derived
from, so check the base model's terms before redistributing — Gemma 4 and Qwen 3.6 are both
Apache-2.0, but that is a property of those releases, not a general rule for open-weight models.
