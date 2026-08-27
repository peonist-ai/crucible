# Crucible

MoE expert compression toolkit — prune, merge, and forge specialist models for local inference.

## Purpose

Take large Mixture-of-Experts models and compress them into compact specialists
that run on consumer hardware. The output is a model file (safetensors → GGUF)
that can be served by Ollama, llama.cpp, or any local inference runtime.

**Any MoE architecture, any compression strategy.** Nothing in the core is
specialised to a particular model: a new architecture is one `ModelAttrs` entry
declaring where its router, experts and projections live, and a new method is one
registry entry. Neither requires a CLI change. The models listed further down are
what the registry currently declares, not the limit of what the method handles.

The shape of a typical run, to make the numbers concrete: a 26B-parameter MoE with
128 experts, pruned 50% and quantized to Q4_K_M, lands near 8 GB — small enough for
a consumer GPU with room left for KV cache. The same pipeline runs on an 8-expert
Mixtral or a 256-expert Qwen; only the registry entry differs.

## Tech Stack

- **Runtime**: Python 3.12, managed with `uv`
- **Core deps**: PyTorch, HuggingFace Transformers, Accelerate, Safetensors
- **Eval**: EleutherAI lm-evaluation-harness
- **Quantization**: llama.cpp (convert → quantize); llm-compressor ≥0.13 for
  compressed-tensors W4A16/FP8 served natively by vLLM; mlx-lm ≥0.31 for the
  Apple-silicon path. All three are optional deps — nothing in the core imports
  them.
- **Serving**: vLLM on ROCm. gfx1151 (Strix Halo) is officially supported as of
  ROCm 7.14 / vLLM 0.27 — prebuilt wheels, no patches, and no
  `HSA_OVERRIDE_GFX_VERSION`. Nothing here builds vLLM from source.

## Architecture

A uv workspace holding two packages. The split is the point: compression
produces weights and needs torch; measurement talks to an HTTP endpoint and
needs neither torch nor transformers. They share **no imports**. The only thing
crossing the boundary is files — crucible writes a model plus
`compression_metadata.json`, crucible-bench writes result JSON.

```
packages/crucible/            ← compression. torch, transformers.
  crucible/
    types.py              ← Core data types (ModelAttrs, MethodContext, etc.)
    cli.py                ← Parser assembly + dispatch, nothing else
    quantize.py           ← compressed-tensors W4A16 contract: full-precision ignore
                            list, calibration windowing, post-save config fixups.
                            llm-compressor is an optional dep, gated in one place.
    quantize_mlx.py       ← MLX contract: bit-allocation policy composed with the
                            model's own predicate (mlx-lm resolves those with a
                            plain `or`, so overriding silently drops MoE router
                            protection), our calibration mix in place of mlx-lm's
                            generic one, and a dry-run size projection.
    quant_types.py        ← ggml type table: exact bpw from the block layouts, plus a
                            MEASURED distortion column. Not monotone — see its docstring.
    gguf.py               ← Read-only GGUF header/directory parser and imatrix reader.
                            Stdlib only; crucible takes no llama.cpp dependency.
    allocate.py           ← Multiple-choice knapsack over (tensor, quant type) under a
                            byte budget, weighted by imatrix activation energy.
    tiers.py              ← Hardware tier presets. Budget+backend are the primitive;
                            a tier just expands to them.
    commands/             ← One module per subcommand; each owns its flags and its run()
      common.py           ← Device/dtype/arch resolution, model loading, score (de)serialization
      plan.py             ← `crucible plan`: chooses a quant type per tensor for a
                            budget or tier, emits a --tensor-type-file. Never runs
                            llama.cpp and never touches weights — it plans, Peonist's
                            Forge builds.
    models/
      registry.py         ← Model adapter registry (maps architectures to MoE internals)
    methods/
      registry.py         ← Method + scorer registry (what --method and --scoring resolve to)
      observer.py         ← Activation collection hooks (expert frequency, norms, routing)
      reap.py             ← REAP: Router-weighted Expert Activation Pruning
      ream.py             ← REAM: Router-weighted Expert Activation Merging

packages/crucible-bench/      ← measurement. datasets + huggingface-hub, nothing else.
  crucible_bench/
    cli.py                ← `crucible-bench run | import | compare | eval-local`
    importers/            ← External harnesses we deliberately don't reimplement
      registry.py         ← What --tool resolves to; one entry per harness
      terminal_bench.py   ← Harbor JobResult → our result JSON
      swebench.py         ← run_evaluation report → our result JSON
    ifeval.py             ← IFEval's 25 checkers, ported from the reference impl
    compare.py            ← Side-by-side comparison (original vs compressed)
    benchmark.py          ← lm-evaluation-harness over local weights (`local` extra)
    testbench/            ← Own-harness benchmarks over an OpenAI-compatible API
      suites.py           ← Task registry (TaskSpec) + named suites
      api.py              ← Endpoint client + response cleanup
      sandbox.py          ← Isolation for model-generated code. Both entry points
                            (execute_code, run_file) contain it the same way; nothing
                            in this package may shell out to model code directly.
      tasks_*.py          ← The benchmarks themselves
    agent_probes/         ← Multi-turn agentic tasks with a real tool loop. Lives here,
                            not in scripts/, because it measures a served endpoint.
                            tasks/ is package data — no __init__.py, it is fixtures.

scripts/                      ← Pipeline tools that feed the compression path: imatrix
                                calibration data, per-role sensitivity, KLD corpus.
                                NOT benchmarks — those belong to crucible-bench.
containers/                   ← bench-sandbox (EvalPlus baseline, pinned) and
                                agent-probe-sandbox (adds pytest). Separate on purpose:
                                extending the baseline would move measured scores.
```

Two registries carry the extension points: `models/registry.py` for a new
architecture, `methods/registry.py` for a new compression or scoring strategy.
A registry entry declares its own CLI flags, so neither needs a CLI edit.

**Do not add benchmarking to `crucible`.** Tests in both packages enforce the
boundary: `crucible` must not expose a benchmark subcommand, and
`crucible_bench` must not import `crucible` or touch torch at module scope. A
benchmark whose grading loop we do not own (Terminal-Bench, SWE-bench) does not
belong in either package — run it with its own harness and import the result.

## Compression Methods

### REAP (Pruning)
Score experts by router gate-values × activation norms. Remove lowest-scoring
experts. Fast, simple, good at moderate compression. Quality degrades at
aggressive ratios because knowledge is discarded.

Reference: arxiv.org/abs/2510.13999

**Upstream note (revised 2026-08-17):** `llm-compressor` ships a `REAPPruningModifier`
(`src/llmcompressor/modifiers/pruning/reap/`, PR #2864) implementing the same paper with the same
saliency metric. Ours predates it and was written independently. Diffed in full; the split is:

- **They only support `LinearExperts2D`** — `utils.py` skips any other experts module with a
  warning, and there is no fused-`gate_up_proj` / 3D-stacked path. Both architectures we have
  validated (Gemma 4, Qwen 3.5/3.6) are `tensor3d` + `fused_gate_up`, so their REAP reaches
  them only after llm-compressor's linearize step — which for Qwen3.5-MoE is the 2D→3D→2D
  round-trip we filed as issue #3037 (~126 GiB to load a 35B-A3B). Granite and Llama4 MoE
  blocks they exclude outright.
- **They handle group-limited routers**, which we now do too (`ModelAttrs.n_group_key` /
  `top_k_group_key`). No registry entry sets them: Qwen3.5-MoE's published config carries no
  `n_group`/`topk_group`, and REAP at 48% was near-lossless on it, so grouping was never being
  violated. The support is for DeepSeek-V3-shaped models.
- **It does not compose with quantization in one pass.** Their `base.py` asserts REAP must be the
  only modifier in the recipe during calibration; prune → quantize is two sequential `oneshot`
  calls. The draw is the output *format* — see `crucible quantize`.
- Ahead on their side beyond grouping: nothing we lack. Ahead on ours: non-uniform per-layer
  budgets, reusable scores across ratios, the alternate scorers, and a GGUF path they have no
  equivalent for.

### REAM (Merge + Prune) — experimental, currently broken
Protect high-saliency experts as centroids. Merge similar neighbors into them
via pseudo-pruning. Sequential layer-by-layer merging with activation recomputation.
Intended to preserve quality better at aggressive compression ratios.

**Status: does not work on Gemma 4** (confirmed 2026-04-14). The merge assumes a
linearity the SiLU-gated experts don't have. Kept because the approach is sound
for architectures that satisfy its assumptions, but REAP is the validated path
and the CLI default. Don't reach for REAM without re-deriving the math first.

Reference: arxiv.org/abs/2604.04356

### Bit allocation (`crucible plan`)
Orthogonal to pruning, and the second half of "best model for this machine".
Given a byte budget it solves which quant type each tensor gets, maximising
accuracy per byte — the piece Unsloth's Dynamic GGUFs have that a hand-written
`--tensor-type` list does not. Two rules it exists to enforce:

- **Plan with an imatrix.** Without one the objective degenerates to
  parameters × distortion, which on an MoE strips attention to feed the expert
  stacks. `crucible plan` refuses rather than emitting that file.
- **Per-expert allocation is unreachable in GGUF.** A layer's experts are one 3D
  tensor and a tensor has one type, however good the per-expert saliency is. The
  finest real granularity is per layer.

### Future
Hybrid approaches, task-specific expert selection, progressive compression with
fine-tuning. The method interface is open — add a new strategy without touching
existing code.

## Evaluation Protocol

Scientific rigor is non-negotiable. Every compression experiment must be
reproducible and comparable.

### Standard Benchmark Suite

Our specialist models are coding agents with tool-calling capability. The
benchmark suite is weighted accordingly — coding and tool use are primary,
general knowledge is secondary (sanity check, not optimization target).

**Primary — Coding:**
- HumanEval / EvalPlus — function-level code generation
- LiveCodeBench — real-world competitive coding
- SWE-bench Lite — repository-level bug fixing (agentic)
- MBPP+ — basic Python programming

**Primary — Tool Use & Instruction Following:**
- IFEval — instruction following fidelity
- BFCL (Berkeley Function Calling Leaderboard) — structured tool/function calling
- Nexus Function Calling — multi-turn tool use

**Primary — Reasoning (code-adjacent):**
- GSM8K (8-shot) — arithmetic reasoning (proxy for code logic)
- MATH-500 — formal reasoning under constraints
- ARC-Challenge (25-shot) — systematic reasoning

**Secondary — General Knowledge (sanity check):**
- MMLU (5-shot) — should not crater; we don't optimize for it
- HellaSwag (10-shot) — commonsense baseline

**Generation Quality (qualitative):**
- Loop/collapse detection — does the model degenerate?
- Output length stability — does compression cause verbosity shifts?
- XML tool-call format compliance — can it still emit structured tool calls?

### Evaluation Rules

1. **Always benchmark the original uncompressed model first** as baseline
2. **Same hardware, same settings** for all variants (temp, top_p, max_tokens)
3. **Record everything**: model ID, method, ratio, calibration data, seed, hardware, eval scores
4. **Results go in `results/`** as structured JSON — one file per run
5. **Compare with tables** — never eyeball, always compute deltas from baseline
6. **No silent passes.** A check the harness cannot actually perform raises —
   it never scores as "passed" and never guesses on the model's behalf. Missing
   dependency, unknown dataset id, malformed ground truth: preflight it and
   refuse to start. Unparseable output is scored wrong *and* recorded
   (`extraction_tier`), so the run summary can report how much of a score rests
   on formatting (`unparsed`) rather than knowledge. Graders are ported from the
   reference implementation checker by checker, not reconstructed from memory —
   an audit on 2026-08-22 found four silent-pass paths, every one of them
   inflating a compressed model's score.

### Calibration Data

The REAM paper showed calibration mix has enormous impact:
- C4 helps MC tasks, hurts generative
- Code helps generative, hurts MC
- Math is neutral

Default calibration mix for coding/tool-use specialists:
- 40% code (evol-codealpaca, SWE-smith trajectories, HumanEval-style)
- 25% tool-calling / agentic (xlam-function-calling, agentic coding traces)
- 20% reasoning (Mixture-of-Thoughts code/math)
- 15% general (C4, QA — prevents catastrophic forgetting on general tasks)

The calibration mix is the biggest lever for specialist quality — larger than
the compression ratio. Different target roles want different mixes: a
research/QA assistant wants more reasoning and general QA, while a coding agent
wants maximum code and tool-calling.

## Supported Models

| Model | Experts | Active | Storage | Status |
|-------|---------|--------|---------|--------|
| Qwen 3.5 / 3.6 35B-A3B | 256 + 1 shared | 8 | tensor3d | validated end-to-end |
| Gemma 4 26B-A4B | 128 | 8 | tensor3d | supported |
| Qwen3 30B-A3B | 128 | 8 | modulelist | supported |
| Mixtral 8x7B | 8 | 2 | modulelist | supported |

This is the registry's current contents, not a limit. Adding an architecture is one
`ModelAttrs` entry in `registry.py`; expert counts from 8 to 256, fused or separate
`gate_up_proj`, `modulelist` or stacked-3D storage, and grouped routing are all
already handled.

## Conventions

- `uv` for dependency management — `uv sync`, `uv run crucible`
- Ruff for linting and formatting
- No classes for core logic — plain functions + dataclasses
- Result types over exceptions at boundaries
- One clear purpose per file
