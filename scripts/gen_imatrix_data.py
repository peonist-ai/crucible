"""Generate calibration text file for llama-imatrix importance matrix computation.

Pulls samples from our coding-specialist calibration mix and writes plain text
suitable for llama-imatrix -f input.

Usage:
    python scripts/gen_imatrix_data.py -o /tmp/imatrix_calibration.txt --samples 256

For a held-out evaluation set drawn from the same distribution, skip past what
calibration consumed:

    python scripts/gen_imatrix_data.py -o /tmp/kld_eval.txt --samples 64 --skip 256

Measuring KL-divergence on the text the imatrix was built from reports how well
the quantization memorised its own calibration set, which is not the question.
Unsloth hold out a separate 300-example set for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import time

from datasets import load_dataset

# ShareGPT-style speaker tags -> chat roles. hermes-function-calling stores
# conversations as {"from": ..., "value": ...} rather than {"role", "content"}.
_ROLE_ALIASES = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    "system": "system",
    "tool": "tool", "function": "tool", "function_response": "tool", "observation": "tool",
}


def extract_messages(sample: dict, field: str) -> list[dict] | None:
    """Normalise a conversation sample to [{"role", "content"}], or None if it is not one.

    Kept separate from `extract_text` because the chat template needs structure —
    once roles are flattened into a single string they cannot be recovered.
    """
    value = sample.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(value, list):
        return None

    out = []
    for msg in value:
        if not isinstance(msg, dict):
            return None
        raw_role = msg.get("role") or msg.get("from") or ""
        content = msg.get("content") or msg.get("value") or ""
        role = _ROLE_ALIASES.get(str(raw_role).strip().lower())
        if role is None or not str(content).strip():
            continue
        out.append({"role": role, "content": str(content)})
    return out or None


def extract_text(sample: dict, field: str) -> str | None:
    """Extract text content from a dataset sample."""
    value = sample.get(field)
    if value is None:
        return None

    # Chat/message format (list of dicts with role/content)
    if isinstance(value, list):
        parts = []
        for msg in value:
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("value") or ""
                parts.append(str(content))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)

    # JSON string containing messages
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return extract_text({"_": parsed}, "_")
        except (json.JSONDecodeError, TypeError):
            pass
        return value

    return str(value)


# Calibration sources: (dataset_id, config, split, field, num_samples)
SOURCES = [
    # Agent coding traces (40%)
    ("open-r1/Mixture-of-Thoughts", "code", "train", "messages", 100),
    ("open-r1/codeforces-cots", None, "train", "messages", 50),
    # Tool/function calling (20%)
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", "conversations", 50),
    # Reasoning (20%)
    ("open-r1/Mixture-of-Thoughts", "math", "train", "messages", 50),
    # General (20% — forgetting anchor)
    ("allenai/c4", "en", "validation", "text", 50),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="/tmp/imatrix_calibration.txt")
    parser.add_argument("--samples", type=int, default=256,
                        help="Total samples (sources are scaled proportionally)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip this many samples per source first, scaled the same "
                             "way as --samples. Use it to draw a held-out set that is "
                             "disjoint from what calibration already consumed.")
    parser.add_argument("--chat-template", default=None, metavar="MODEL_DIR",
                        help="Render conversations through this model's chat template "
                             "instead of flattening them to bare text. An imatrix "
                             "measures activation statistics, so calibration that omits "
                             "the role markers and special tokens every real prompt "
                             "carries measures them off-distribution. Plain-text sources "
                             "(the C4 anchor) are passed through untouched.")
    parser.add_argument("--exclude-file", default=None, metavar="PATH",
                        help="Drop samples whose text overlaps this file. --skip gives "
                             "disjoint SAMPLES, not disjoint CONTENT: these datasets "
                             "carry the same problem more than once with different "
                             "reasoning traces, so a skipped-past duplicate can "
                             "reappear. Point this at the calibration file when "
                             "building an eval set.")
    parser.add_argument("--retries", type=int, default=3, metavar="N",
                        help="Attempts per source before giving up (default: 3).")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Continue when a source fails or runs short. Off by "
                             "default: a missing source silently changes the mix, and "
                             "the mix is the single biggest lever on specialist quality.")
    args = parser.parse_args()

    tokenizer = None
    if args.chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
        if not getattr(tokenizer, "chat_template", None):
            raise SystemExit(f"{args.chat_template} has no chat_template to apply")
        print(f"  Rendering conversations through {args.chat_template}")

    # Long lines only: short ones are shared boilerplate (fences, role markers,
    # a dataset's standing instruction preamble) and would reject everything.
    excluded_lines: set[str] = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            excluded_lines = {ln.strip() for ln in f if len(ln.strip()) > 120}
        print(f"  Excluding content overlapping {args.exclude_file} "
              f"({len(excluded_lines)} distinctive lines)")

    templated = flattened = dropped_overlap = 0
    total = sum(s[4] for s in SOURCES)
    scale = args.samples / total
    skip_scale = args.skip / total
    all_texts = []
    shortfalls = []

    for ds_id, config, split, field, count in SOURCES:
        n = max(1, int(count * scale))
        skip_n = int(count * skip_scale)
        label = f"{ds_id} ({config or 'default'})"
        skipped = f", after skipping {skip_n}" if skip_n else ""
        print(f"  Loading {label}... {n} samples{skipped}")
        collected = 0
        last_error: Exception | None = None
        # Streamed reads fail transiently often enough to matter — connection
        # resets, and a client-lifecycle bug in `datasets` that surfaces as
        # "Cannot send a request, as the client has been closed". Retrying the
        # source is right; failing the whole mix over a dropped socket is not.
        for attempt in range(1, args.retries + 1):
            try:
                ds = load_dataset(ds_id, config, split=split, streaming=True)
                seen = collected = 0
                before = len(all_texts)
                for sample in ds:
                    if collected >= n:
                        break
                    text, was_templated = None, False
                    if tokenizer is not None:
                        messages = extract_messages(sample, field)
                        if messages:
                            try:
                                text = tokenizer.apply_chat_template(messages, tokenize=False)
                                was_templated = True
                            except Exception:
                                text = None  # fall through to the flat form
                    if text is None:
                        text = extract_text(sample, field)
                    if not text or len(text) <= 100:
                        continue
                    seen += 1
                    if seen <= skip_n:
                        continue      # counted only once kept, so the tally matches the file
                    if excluded_lines and any(
                        ln.strip() in excluded_lines
                        for ln in text.splitlines()
                        if len(ln.strip()) > 120
                    ):
                        dropped_overlap += 1
                        continue
                    all_texts.append(text)
                    collected += 1
                    if was_templated:
                        templated += 1
                    else:
                        flattened += 1
                last_error = None
                break
            except Exception as e:
                last_error = e
                del all_texts[before:]        # drop this attempt's partial take
                collected = 0
                print(f"    attempt {attempt}/{args.retries} failed: {e}")
                if attempt < args.retries:
                    time.sleep(5 * attempt)

        if last_error is not None:
            shortfalls.append(f"{label}: {type(last_error).__name__}: {last_error}")
            continue
        print(f"    Got {collected}/{n}")
        if collected < n:
            shortfalls.append(f"{label}: {collected} of {n}")

    if shortfalls and not args.allow_partial:
        raise SystemExit(
            "calibration mix is not what was asked for:\n  "
            + "\n  ".join(shortfalls)
            + "\n\nThe mix drives specialist quality more than the compression ratio does, "
              "so a quietly reweighted one invalidates the run.\nRe-run with "
              "--allow-partial only if you have decided the shortfall does not matter."
        )

    with open(args.output, "w") as f:
        for text in all_texts:
            # llama-imatrix expects plain text, one doc per line or separated
            f.write(text.strip() + "\n")

    total_chars = sum(len(t) for t in all_texts)
    print(f"\n  Wrote {len(all_texts)} samples ({total_chars/1e6:.1f}M chars) to {args.output}")
    if tokenizer is not None:
        print(f"  {templated} rendered through the chat template, {flattened} left as plain text")
    if dropped_overlap:
        print(f"  Dropped {dropped_overlap} samples overlapping {args.exclude_file}")


if __name__ == "__main__":
    main()
