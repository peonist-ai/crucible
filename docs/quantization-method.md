# How Crucible allocates bits

Notes from building the REAP-48 v2 GGUF, 2026-08-25. Written down because most of it is
counter-intuitive and two of the findings cost hours to discover.

## The short version

At a fixed file size, a GGUF's quality is decided by *where* the bits go. `crucible plan` solves
that allocation as a multiple-choice knapsack:

```
minimise   Σ  rmse(type)² × n_params(group) × energy_norm(group) × sensitivity(role)
subject to Σ  bytes ≤ budget
```

- `rmse(type)` — **measured**, not modelled. Round-trip error through `ggml_quantize_chunk`
  itself over 1M synthetic weights. See `crucible/quant_types.py`.
- `energy_norm` — imatrix activation energy, normalised **within each role**.
- `sensitivity` — measured per-role KL-divergence. See `scripts/measure_role_sensitivity.py`.

Solved greedily on marginal gain (Δerror ÷ Δbytes) with each group's rungs pruned to their lower
convex hull, which makes greedy exact for the continuous relaxation.

## Finding 1: imatrix energy is not comparable across roles

This is the big one, and getting it wrong produces a model that is confidently broken.

An importance matrix records how large a tensor's **inputs** are. On a normalised transformer
that clusters by *position relative to the nearest RMSNorm*, not by importance. Measured on
Qwen 3.6 REAP-48 at layer 20:

| E[x²] | tensors | what they read |
|---|---|---|
| 0.9218 | `attn_qkv`, `attn_gate`, `ssm_alpha` | the input-normed hidden state |
| 0.8927 | `ffn_gate_exps`, `ffn_gate_shexp` | the post-attention-normed hidden state |
| 0.0110 | `ffn_down_exps`, `ffn_down_shexp` | the FFN intermediate |
| 2.78e-4 | `ssm_out` | the delta-net output |

Anything reading a normed hidden state measures ~0.9 *by construction*. Compare those raw and a
solver concludes `ssm_out` does not matter — it assigned it **1.75 bpw**, which destroys the model.

**The obvious fix does not work.** The dropped `σ_W²` term looks like the culprit; it is not.
Measured, weight RMS spans only 2.4× (0.0086–0.021) across these same tensors, and including σ_W²
*widens* the spread from 3,317× to 4,811×. The signal is simply not in the imatrix.

So energy is used only **within** a role — "which layers of this role run hotter than their
siblings", where the graph-position artefact cancels. Cross-role allocation needs a different
measurement entirely.

## Finding 2: the imatrix is *inverted* on the most important role

Measured per-role ΔKLD (hold all roles at Q6_K, drop one to Q3_K, measure against f16):

| role | imatrix rank | measured rank (total ΔKLD) |
|---|---|---|
| `ssm_out` | **18th of 18** (least) | **1st of 18** (most) |

Not merely uninformative — actively wrong about the single highest-impact role in the model.

Most sensitive per parameter: `attn_v` (0.374 per 1B params, and only 10.5M params — lifting it
to Q8_0 costs ~6.6 MB), `attn_k` (0.133), `ssm_alpha` (0.108). Least: the routed experts at
~0.0011, which is 76% of all parameters. Spending fewest bits on the experts is correct, and now
measured rather than assumed.

## Finding 3: measurement alone lost to good priors

The uncomfortable result. Four builds, all at 8.78 GiB, scored on KL-divergence against f16:

| build | allocation | Mean KLD vs shipped |
|---|---|---|
| shipped v1 | hand recipe, untemplated imatrix | — |
| v3 | hand structural floors + per-layer imatrix | −42.0% |
| v4 | **measured sensitivity, no floors** | −37.4% |
| **v2 release** | floors **+** measured sensitivity | **−42.5%** |

Measured sensitivity *alone* lost to hand-written floors by 7.8% mean KLD — while being 31%
better on max KLD. Two causes, both in the objective rather than the measurement:

1. **Measured at the wrong operating point.** The sweep used Q6_K→Q3_K; the plan runs near
   3.9 bpw. Sensitivity in a high-precision regime does not transfer cleanly.
2. **Extreme coefficients meet a linear objective.** `attn_v` normalises to 92.6× the mean, and
   error linear in `rmse²` over-concentrates. The uniform floors were accidentally *damping* that.

**Better information fed into a flawed objective can lose to a prior that compensates for the
flaw.** Combining them wins on both average and tail, which is what the release ships.

## Finding 4: calibration must carry the chat template

The largest single contributor to the improvement, and the least clever. The calibration
generator was flattening conversations to `"\n".join(content)` — discarding every `<|im_start|>`,
every role marker, every thinking tag. An imatrix measures activation statistics, so this measured
them on text the model never sees. `--chat-template MODEL_DIR` fixes it.

## Two traps that silently produce wrong numbers

**`n_batch != n_ubatch` corrupts quantized GatedDeltaNet models.** PPL 1402.95 at llama.cpp's
default `-b 2048 -ub 512` versus 2.72 at `-b 2048 -ub 2048`, same model, same input. f16 is
unaffected. The model *generates* perfectly coherent text throughout, because decode never splits
a ubatch. Always pass matching `-b`/`-ub` for perplexity and KLD runs.

**A KL-divergence baseline costs ~0.5 MB per token.** `llama-perplexity` stores
`2*((n_vocab+1)/2)+4` uint16s per token; at a 248,320 vocab a 411K-token corpus writes **204 GB**.
Cap with `--chunks`. And because `--chunks` takes a prefix, interleave a multi-source corpus or
truncation silently drops whole sources.

## Evaluation corpus

Divergence is measured on the **benchmark prompts themselves** — HumanEval+ (164), MBPP+ (378),
BigCodeBench (1140), rendered through the chat template, sources round-robin interleaved.
Measured **0 lines shared** with the calibration set.

The alternative — a held-out slice of the calibration datasets — leaks: those datasets carry the
same problem more than once with different reasoning traces, so skipping past a sample does not
skip its content (measured: 24 shared lines). Benchmark prompts are disjoint by construction and
on-distribution for the task the model is compressed to serve.
