"""The compressed-tensors output contract.

Pruning and quantization are two separate one-shot passes over the same model,
and this module owns the second one's shared pieces: which modules must stay at
full precision, how the calibration corpus is windowed, and the post-save config
fixups a served checkpoint needs.

Kept out of `commands/quantize.py` so the pieces are importable without paying
for the CLI, and out of `scripts/` because the ignore list is derived from the
model registry — it is library logic that happens to be used by a script.

llm-compressor is an optional dependency. Nothing here imports it at module
scope; `require_llmcompressor()` is the single gate, so a missing install
produces one clear message instead of an ImportError from the middle of a run.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.models.registry import MODEL_REGISTRY

_INSTALL_HINT = (
    'llm-compressor is not installed. Install it with:\n'
    '    uv pip install "llmcompressor>=0.13" "transformers>=5.9"\n'
    "It is an optional dependency — only `crucible quantize` needs it."
)


@dataclass
class LLMCompressor:
    """The handful of llm-compressor entry points this command uses."""

    oneshot: Any
    load_context: Any


def require_llmcompressor() -> LLMCompressor:
    """Import llm-compressor, or fail with an actionable message."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.utils import load_context
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise SystemExit(f"{_INSTALL_HINT}\n\nUnderlying error: {e}") from e
    return LLMCompressor(oneshot=oneshot, load_context=load_context)


def quant_ignore_patterns(
    keep_attention: bool = True, keep_shared_expert: bool = True
) -> list[str]:
    """Modules that must stay full precision, across every supported architecture.

    Routers pick which experts fire; quantizing one corrupts routing for every
    token that passes through it. Their names differ per family — `router` on
    Gemma 4, `mlp.gate` on Qwen3/3.5/3.6, `block_sparse_moe.gate` on Mixtral —
    so derive the patterns from the same registry the compression methods use
    instead of hardcoding one family's convention. Shared-expert gates go too:
    they are 1-output Linears that scale a path measured at ~69% load-bearing on
    Qwen 3.6.

    `keep_attention` holds the attention projections at full precision. On an MoE
    this is nearly free on size and is where the quality risk concentrates: for
    Ornith the experts are 32.21B of ~35B params, so all attention together is
    ~1.19B — under 4% of the model. Trading ~1.8GB of size for uncompressed
    attention is a good deal when size is not the binding constraint. It is not
    free on *speed*, though: attention is dense (every token reads all of it)
    while only 8/256 experts fire, so at batch-1 decode attention is roughly half
    the weight traffic. Expect to give up part of the int4 speedup — and note the
    corollary measured on gfx1151: holding every dense Linear at BF16 leaves the
    RDNA-tuned W4A16 kernel with nothing to do, so the quant buys capacity (4.3x
    KV cache) rather than throughput.

    `keep_shared_expert` follows the same logic and is cheaper still: the shared
    expert is 3 x hidden x shared_intermediate per layer (~126M params, 0.4% of
    the model) on a path measured at ~69% load-bearing. The *routed* experts are
    always quantized — they are the entire point.
    """
    patterns = {"lm_head"}
    for attrs in MODEL_REGISTRY.values():
        for path in (attrs.router, attrs.shared_expert_gate):
            if path:
                patterns.add(f"re:.*{re.escape(path)}$")
    if keep_attention:
        # Both attention flavours on the Qwen3.5/3.6 hybrid stack: full-attention
        # layers (self_attn.{q,k,v,o}_proj) and gated-delta-net linear-attention
        # layers (linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}).
        # self_attn also covers Gemma 4 / Mixtral / Qwen3.
        patterns.add(r"re:.*self_attn\..*")
        patterns.add(r"re:.*linear_attn\..*")
    if keep_shared_expert:
        patterns.add(r"re:.*shared_expert\..*")
    return sorted(patterns)


def build_recipe(algorithm: str, ignore: list[str]) -> list:
    """Build the llm-compressor recipe for a W4A16 quantization.

    W4A16 for MoE: 4-bit weights, 16-bit activations, symmetric — vLLM has no
    asym MoE. Format is compressed-tensors for all three algorithms, which is
    what the RDNA-tuned kernel needs; the algorithm only affects quality.

    `gptq` is the default because AWQ is BROKEN on Qwen3.5/3.6-family
    hybrid-attention models as of llm-compressor 0.13.0. AWQ's `_apply_smoothing`
    re-runs the *parent* block to compare fp16 vs scaled outputs, and the kwargs
    it cached don't match the gated-delta-net signature:
        TypeError: Qwen3_5MoeGatedDeltaNet.forward() missing 1 required
                   positional argument: 'hidden_states'
    GPTQ has no such step — it quantizes each Linear from that layer's own
    inputs — so it works here, and it is the same family as the AutoRound
    checkpoint validated on this architecture.
    """
    from llmcompressor.modifiers.quantization import (
        GPTQModifier,
        QuantizationModifier,
    )

    if algorithm == "gptq":
        return [GPTQModifier(scheme="W4A16", targets=["Linear"], ignore=ignore)]
    if algorithm == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            AWQModifier(),
            QuantizationModifier(
                scheme="W4A16", targets=["Linear"], ignore=ignore
            ),
        ]
    if algorithm == "rtn":
        # Round-to-nearest, no calibration search. Fastest, lowest quality.
        return [
            QuantizationModifier(scheme="W4A16", targets=["Linear"], ignore=ignore)
        ]
    raise ValueError(f"Unknown algorithm '{algorithm}'")


@dataclass
class CalibrationWindows:
    """A windowed calibration corpus, plus the record of how it was built."""

    dataset: Any
    collator: Any
    texts: list[str]
    stats: dict


def build_calibration_windows(
    tokenizer,
    *,
    profile: str,
    num_samples: int,
    max_seq_length: int,
    seed: int,
) -> CalibrationWindows:
    """Tokenize the calibration mix and split it into fixed-size windows.

    Chunk instead of truncate. Rendered agentic traces are long: measured on the
    `default` profile with Ornith's tokenizer, median 13,171 tokens, p95 52,678,
    max 176,967. Letting the tokenizer truncate at 8192 discarded 65% of every
    sample, and the discarded part is never random — it is always the *tail*.
    That biases calibration toward conversation openings (long system prompts and
    tool definitions) and away from the later tool-call/response turns a coding
    agent actually spends its time on. Raising max_seq_length cannot fix it: the
    tail reaches 177K tokens, and eager attention is O(S^2) in memory.

    So: tokenize whole, split into windows, shuffle the windows, and take
    `num_samples` of them. Nothing is discarded for being late in a conversation,
    and windows are drawn from every position. `num_samples` counts *windows*,
    which is also the honest unit — it is what drives both runtime and
    activation-cache size.

    Sharing `build_calibration_texts()` with the observer matters: quantizing on
    a different distribution than the one the expert scores were derived from
    would be a silent confound.
    """
    import torch
    from datasets import Dataset

    from crucible.data import build_calibration_texts

    texts = build_calibration_texts(
        tokenizer, profile=profile, num_samples=num_samples, seed=seed
    )

    rng = random.Random(seed)
    windows: list[list[int]] = []
    raw_lengths: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        raw_lengths.append(len(ids))
        for start in range(0, len(ids), max_seq_length):
            chunk = ids[start:start + max_seq_length]
            # A stub tail carries little signal and still costs a full forward.
            if len(chunk) >= min(512, max_seq_length):
                windows.append(chunk)
    rng.shuffle(windows)
    selected = windows[:num_samples]
    if not selected:
        raise ValueError(
            "Calibration produced no usable windows — every sample was shorter "
            f"than the {min(512, max_seq_length)}-token floor."
        )

    dataset = Dataset.from_dict({
        "input_ids": selected,
        "attention_mask": [[1] * len(w) for w in selected],
    })

    def collator(batch):
        assert len(batch) == 1, "sequential pipeline calibrates one sample at a time"
        return {k: torch.tensor(v).unsqueeze(0) for k, v in batch[0].items()}

    lengths = sorted(raw_lengths)
    win_lengths = sorted(len(w) for w in selected)
    stats = {
        "profile": profile,
        "seed": seed,
        "max_seq_length": max_seq_length,
        "source_conversations": len(texts),
        "conv_token_median": lengths[len(lengths) // 2],
        "conv_token_p95": lengths[int(len(lengths) * 0.95)],
        "conv_token_max": lengths[-1],
        "windows_available": len(windows),
        "windows_requested": num_samples,
        "windows_used": len(selected),
        "window_token_median": win_lengths[len(win_lengths) // 2],
        "total_calibration_tokens": sum(win_lengths),
        "tokens_truncated": 0,
        "coverage_pct": round(100 * len(selected) / max(1, len(windows)), 2),
    }
    return CalibrationWindows(
        dataset=dataset, collator=collator, texts=texts, stats=stats
    )


def realign_ignore_list(config_path: str | Path) -> int:
    """Rewrite the saved ignore list to match the saved tensor names.

    For a multimodal wrapper checkpoint the modules are named
    `model.language_model.layers.N...` in memory, so that is what llm-compressor
    records in quantization_config.ignore. But transformers' PrefixChange strips
    `language_model.` on the way to disk, so the tensors land as
    `model.layers.N...`. vLLM's `should_ignore_layer` does EXACT string matching
    (regex only for `re:`-prefixed entries), so every attention / shared-expert
    entry would silently fail to match — vLLM would then treat those layers as
    quantized, look for `weight_packed`, and find only BF16 `weight`.

    Returns the number of entries rewritten.
    """
    path = Path(config_path)
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    ignore = config.get("quantization_config", {}).get("ignore")
    if not ignore:
        return 0
    fixed = [name.replace("model.language_model.", "model.") for name in ignore]
    changed = sum(1 for a, b in zip(ignore, fixed) if a != b)
    if changed:
        config["quantization_config"]["ignore"] = fixed
        path.write_text(json.dumps(config, indent=2))
    return changed
