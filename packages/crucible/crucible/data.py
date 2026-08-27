"""Calibration data loading for expert compression.

Fetches datasets from HuggingFace, tokenizes them, and returns a
DataLoader with the configured mix ratios. Supports raw text datasets,
instruction pairs, and chat-formatted message datasets.

The calibration mix is the biggest quality lever in MoE compression.
Datasets here are chosen to avoid eval benchmark contamination.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer

# Each entry: (dataset_id, config, split, text_extractor, ratio)
# config=None means no subset. text_extractor is a field name or "messages".
#
# Design rationale:
#   - No eval benchmarks (gsm8k, humaneval, MATH-500) to avoid contamination
#   - Heavy code+tool weight (55%) for coding specialist use case
#   - Mixture-of-Thoughts for reasoning without benchmark leakage
#   - C4 as general anchor to prevent catastrophic forgetting

CalibrationSource = tuple[str, str | None, str, str, float]

# Calibration mix v3 — coding agent specialist
#
# Optimized for: coding agents with tool-calling capability
# Based on: REAM paper correlations (C4 r=-0.82 with GEN, code r=+0.59)
#
# Priorities:
#   1. Real agent traces (most representative of deployment)
#   2. Code generation + reasoning
#   3. Tool/function calling
#   4. General reasoning (code logic proxy)
#   5. Minimal general text (forgetting anchor only)
#
# Calibration profiles — named presets for different scoring strategies.
# "default" is the original coding-specialist mix.
# "code-only" is pure code for task-focused scoring.
#
# Each entry: (dataset_id, config, split, text_extractor, ratio)

# Gold-only coding-agent mix. Every dataset has been probed and verified:
#   - structured multi-turn messages with proper role/content
#   - modern <tools>/<tool_call> format (not legacy <functioncall>)
#   - open access (no gating)
#
# Coverage: 60% agent/tool-use, 25% code reasoning, 10% math reasoning, 5% general.
#
# Dropped from prior mixes (quality issues — see project_qwen36_experiments.md):
#   - glaiveai/glaive-function-calling-v2: flat string format, old <functioncall> syntax
#   - nlile/misc-merged-claude-code-traces-v1: often user-only prompts
#   - AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset: flat blob in assistant field
_MIX_DEFAULT: list[CalibrationSource] = [
    # --- Real agent coding + tool use (60%) ---
    ("SWE-bench/SWE-smith-trajectories", None, "tool", "messages", 0.25),
    ("lambda/hermes-agent-reasoning-traces", "kimi", "train", "conversations", 0.20),
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "conversations", 0.15),
    # --- Code reasoning (25%) ---
    ("open-r1/Mixture-of-Thoughts", "code", "train", "messages", 0.15),
    ("open-r1/codeforces-cots", None, "train", "messages", 0.10),
    # --- Math / formal reasoning (10%) ---
    ("open-r1/Mixture-of-Thoughts", "math", "train", "messages", 0.10),
    # --- General (5%) — catastrophic forgetting anchor ---
    ("allenai/c4", "en", "train", "text", 0.05),
]

# 100% code — for task-focused scoring where we only care about
# coding performance. No general text, no reasoning prose, no tool-calling.
# Scores experts purely by their contribution to code generation.
#
# NO eval benchmark data (HumanEval, MBPP test) to avoid contamination.
# MBPP train split is safe — separate from the test split we benchmark on.
_MIX_CODE_ONLY: list[CalibrationSource] = [
    # MBPP train split — Python functions with solutions (no test overlap)
    ("mbpp", "full", "train", "code", 0.25),
    # Pure code outputs (Python implementations)
    ("theblackcat102/evol-codealpaca-v1", None, "train", "output", 0.35),
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train", "output", 0.40),
]

# General-only — for contrast scoring (what's NOT code-specific)
_MIX_GENERAL: list[CalibrationSource] = [
    ("allenai/c4", "en", "train", "text", 0.50),
    ("wikimedia/wikipedia", "20231101.en", "train", "text", 0.50),
]

CALIBRATION_PROFILES: dict[str, list[CalibrationSource]] = {
    "default": _MIX_DEFAULT,
    "code-only": _MIX_CODE_ONLY,
    "general": _MIX_GENERAL,
}

# Backwards compat
DEFAULT_CALIBRATION_MIX = _MIX_DEFAULT


def build_calibration_texts(
    tokenizer: PreTrainedTokenizer,
    datasets: list[str] | None = None,
    profile: str | None = None,
    num_samples: int = 1024,
    seed: int = 42,
) -> list[str]:
    """Build calibration samples as chat-template-rendered text.

    Shared by the activation observer (through build_calibration_dataloader,
    which tokenizes these) and by one-shot quantizers such as llm-compressor,
    which want text rather than tensors. One implementation means both paths
    see the same mix and the same tool-call rendering — a quantizer calibrated
    on a different distribution than the scorer would be a silent confound.

    Args:
        tokenizer: model tokenizer.
        datasets: list of HuggingFace dataset IDs. If None, uses
            profile or DEFAULT_CALIBRATION_MIX.
        profile: named calibration profile ("default", "code-only", "general").
            Ignored if datasets is set.
        num_samples: total number of calibration samples.
        seed: random seed for reproducibility.

    Returns:
        Rendered strings, one per calibration sample.
    """
    import datasets as hf_datasets

    hf_datasets.logging.set_verbosity_error()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_samples: list[list[dict]] = []

    if datasets is None:
        mix = (
            CALIBRATION_PROFILES.get(profile, DEFAULT_CALIBRATION_MIX)
            if profile
            else DEFAULT_CALIBRATION_MIX
        )
        for ds_id, config, split, field, ratio in mix:
            n = max(1, int(num_samples * ratio))
            print(f"    {ds_id} ({config or split}): {n} samples...")
            samples = _load_texts(ds_id, config, split, field, n, seed)
            print(f"      loaded {len(samples)}")
            all_samples.extend(samples)
    else:
        per_ds = max(1, num_samples // len(datasets))
        for ds_id in datasets:
            config, split, field = _guess_dataset_params(ds_id)
            print(f"    {ds_id}: {per_ds} samples...")
            samples = _load_texts(ds_id, config, split, field, per_ds, seed)
            print(f"      loaded {len(samples)}")
            all_samples.extend(samples)

    # Shuffle and cap
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_samples), generator=gen).tolist()
    all_samples = [all_samples[i] for i in indices[:num_samples]]

    # Render each sample through the chat template with its real role structure.
    # Multi-turn conversations keep <|im_start|>user / <|im_start|>assistant
    # blocks so routing activations match what the model sees at inference
    # time. Single-turn samples are already wrapped as one user message by the
    # extractor, so this path handles both uniformly.
    all_texts: list[str] = []
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        for messages in all_samples:
            # Only add a generation prompt for single-turn user-only samples.
            # Multi-turn conversations already contain assistant turns, so
            # appending another <|im_start|>assistant would produce a dangling
            # empty assistant turn.
            is_single_turn_user = (
                len(messages) == 1 and messages[0].get("role") == "user"
            )
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=is_single_turn_user,
                )
            except Exception:
                rendered = _plain_render(_flatten_tool_calls(messages))

            # Templates that don't understand tool_calls drop them silently —
            # no exception, just a shorter string missing every tool call the
            # agent made. Detect that and re-render with them folded into
            # content, so tool-calling activations survive on any model.
            if _tool_calls_missing(messages, rendered):
                flat = _flatten_tool_calls(messages)
                try:
                    rendered = tokenizer.apply_chat_template(
                        flat,
                        tokenize=False,
                        add_generation_prompt=is_single_turn_user,
                    )
                except Exception:
                    rendered = _plain_render(flat)
            all_texts.append(rendered)
    else:
        # No chat template — fall back to plain role-prefixed concatenation.
        all_texts = [
            _plain_render(_flatten_tool_calls(messages)) for messages in all_samples
        ]

    return all_texts


def build_calibration_dataloader(
    tokenizer: PreTrainedTokenizer,
    datasets: list[str] | None = None,
    profile: str | None = None,
    num_samples: int = 1024,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Build a calibration DataLoader from HuggingFace datasets.

    Thin wrapper: build_calibration_texts() picks and renders the mix, this
    tokenizes it.

    Args:
        tokenizer: model tokenizer (must have pad_token set).
        datasets: list of HuggingFace dataset IDs. If None, uses
            profile or DEFAULT_CALIBRATION_MIX.
        profile: named calibration profile ("default", "code-only", "general").
            Ignored if datasets is set.
        num_samples: total number of calibration samples.
        max_seq_length: max token length per sample.
        batch_size: batch size for the DataLoader.
        seed: random seed for reproducibility.

    Returns:
        DataLoader yielding dicts with 'input_ids' and 'attention_mask'.
    """
    all_texts = build_calibration_texts(
        tokenizer,
        datasets=datasets,
        profile=profile,
        num_samples=num_samples,
        seed=seed,
    )

    encodings = tokenizer(
        all_texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": masks}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )


def _load_texts(
    dataset_id: str,
    config: str | None,
    split: str,
    field: str,
    num_samples: int,
    seed: int,
) -> list[list[dict]]:
    """Load calibration samples as lists of chat messages.

    Returns a list of message lists (each inner list is one sample ready for
    tokenizer.apply_chat_template). Preserves role structure for multi-turn
    conversations so the tokenizer renders proper <|im_start|>role blocks.
    """
    import datasets as hf_datasets

    try:
        ds = hf_datasets.load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
        )
    except Exception:
        # Fallback: try without config
        ds = hf_datasets.load_dataset(
            dataset_id,
            split=split,
            streaming=True,
        )

    ds = ds.shuffle(seed=seed)

    samples: list[list[dict]] = []
    for example in ds:
        messages = _extract_messages(example, field)
        if messages is None:
            continue
        if _message_text_len(messages) < 50:
            continue
        samples.append(messages)
        if len(samples) >= num_samples:
            break

    return samples


def _plain_render(messages: list[dict]) -> str:
    """Role-prefixed concatenation, used when no chat template applies."""
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)


def _tool_calls_missing(messages: list[dict], rendered: str) -> bool:
    """True if the sample has tool calls the rendered text doesn't contain."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = fn.get("name")
            if name and name not in rendered:
                return True
    return False


def _tool_calls_to_text(tool_calls: list) -> str:
    """Flatten tool calls to text, for chat templates that can't render them.

    Deliberately model-neutral JSON rather than `<tool_call>` tags: this path
    only runs when the template has no native tool-call syntax, so inventing
    another model's markup would inject tokens this model never emits.
    """
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or ""
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        if name or args:
            parts.append(f'{{"name": "{name}", "arguments": {args or "{}"}}}')
    return "\n".join(parts)


def _message_text_len(messages: list[dict]) -> int:
    """Total usable calibration text in a sample, tool calls included."""
    total = 0
    for m in messages:
        total += len(m.get("content", ""))
        if m.get("tool_calls"):
            total += len(_tool_calls_to_text(m["tool_calls"]))
    return total


def _flatten_tool_calls(messages: list[dict]) -> list[dict]:
    """Fold tool_calls into content so any renderer preserves them."""
    flattened = []
    for m in messages:
        if not m.get("tool_calls"):
            flattened.append(m)
            continue
        rendered = _tool_calls_to_text(m["tool_calls"])
        content = m.get("content", "")
        merged = f"{content}\n{rendered}".strip() if content else rendered
        flattened.append({"role": m.get("role", "assistant"), "content": merged})
    return flattened


def _extract_messages(example: dict, field: str) -> list[dict] | None:
    """Extract structured messages from a dataset example.

    Always returns a list of {"role", "content"} dicts so the tokenizer's
    chat template renders the conversation with proper role markers. This
    matters for MoE models where different roles activate different experts —
    the prior version flattened multi-turn conversations to a single string
    and re-wrapped it as one user message, missing the assistant-side experts
    that specialized agent/tool models use for structured output generation.

    Handles:
      - Simple string fields ("text", "output", "chat") → single user message
      - JSON string fields (messages_json — parse first)
      - OpenAI message lists ([{"role": ..., "content": ...}, ...])
      - ShareGPT message lists ([{"from": ..., "value": ...}, ...])
      - Assistant turns whose payload is in `tool_calls` with empty `content`,
        as modern agent-trace datasets emit. These are kept with `tool_calls`
        attached so the chat template can render them natively; dropping them
        would lose the tool-calling activations entirely *and* break role
        alternation, leaving user → tool → tool sequences.

    Returns None when the example is empty / unusable, which lets the
    caller skip it.
    """
    val = example.get(field, "")

    # JSON string — parse it first (e.g., messages_json from claude traces)
    if isinstance(val, str) and val.startswith("["):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            pass  # fall through and treat as plain string

    # Structured messages
    if isinstance(val, list) and val and isinstance(val[0], dict):
        messages: list[dict] = []
        for msg in val:
            tool_calls = msg.get("tool_calls") or None
            # OpenAI format: {"role": "user", "content": "..."}
            if "content" in msg:
                role = msg.get("role") or "user"
                content = msg.get("content") or ""
            # ShareGPT format: {"from": "gpt", "value": "..."}
            elif "value" in msg:
                sg_from = msg.get("from", "").lower()
                role = _normalize_sharegpt_role(sg_from)
                content = msg.get("value") or ""
            elif tool_calls:
                role = msg.get("role") or "assistant"
                content = ""
            else:
                continue
            if isinstance(content, list):
                # Multi-part content (e.g. mm with text+image) — concatenate text parts
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if not content and not _tool_calls_to_text(tool_calls or []):
                # Nothing usable — skip rather than emit a dangling empty turn.
                continue
            out = {"role": role, "content": str(content)}
            if tool_calls:
                out["tool_calls"] = tool_calls
            # Some chat templates key tool results off these.
            for passthrough in ("name", "tool_call_id"):
                if msg.get(passthrough):
                    out[passthrough] = msg[passthrough]
            messages.append(out)
        return messages or None

    # Plain list of strings — join and wrap as user message
    if isinstance(val, list):
        text = " ".join(str(v) for v in val if v)
        return [{"role": "user", "content": text}] if text else None

    # Plain string — wrap as user message
    text = str(val) if val else ""
    return [{"role": "user", "content": text}] if text else None


def _normalize_sharegpt_role(sg_from: str) -> str:
    """Map ShareGPT role names to OpenAI-style roles."""
    mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "chatgpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
        "observation": "tool",
    }
    return mapping.get(sg_from, "user")


# Known dataset parameters for common datasets
_KNOWN_DATASETS: dict[str, tuple[str | None, str, str]] = {
    # (config, split, field)
    "nlile/misc-merged-claude-code-traces-v1": (None, "train", "messages_json"),
    "lambda/hermes-agent-reasoning-traces": ("kimi", "train", "conversations"),
    "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset": (None, "train", "assistant"),
    "theblackcat102/evol-codealpaca-v1": (None, "train", "output"),
    "glaiveai/glaive-function-calling-v2": (None, "train", "chat"),
    "open-r1/Mixture-of-Thoughts": ("all", "train", "messages"),
    "open-r1/codeforces-cots": (None, "train", "messages"),
    "SWE-bench/SWE-smith-trajectories": (None, "tool", "messages"),
    "NousResearch/hermes-function-calling-v1": ("func_calling", "train", "conversations"),
    "allenai/c4": ("en", "train", "text"),
    "wikitext": (None, "train", "text"),
    "codeparrot/github-code": (None, "train", "code"),
    "bigcode/starcoderdata": (None, "train", "content"),
    "microsoft/orca-math-word-problems-200k": (None, "train", "question"),
    "wikimedia/wikipedia": ("20231101.en", "train", "text"),
    "emozilla/pg19": (None, "train", "text"),
}


def _guess_dataset_params(dataset_id: str) -> tuple[str | None, str, str]:
    """Guess config, split, and text field for a dataset."""
    if dataset_id in _KNOWN_DATASETS:
        return _KNOWN_DATASETS[dataset_id]
    return (None, "train", "text")
