"""Build a KL-divergence corpus from the benchmark prompts themselves.

The alternative to a proxy corpus. Measuring divergence on held-out slices of the
*calibration* datasets answers "does the quantized model still track the original
on text like what it was calibrated on" — which is a weaker question than the one
we care about, and one that leaks: these datasets carry the same problem more than
once with different reasoning traces, so skipping past a sample does not guarantee
skipping past its content.

The benchmark prompts have neither problem. They are disjoint from the calibration
mix by construction (that mix is Mixture-of-Thoughts, codeforces-cots, hermes
function-calling and C4 — none of them HumanEval, MBPP or BigCodeBench), and they
are the exact distribution the model is being compressed to serve. Divergence
measured here is divergence on the target task.

Prompts are reproduced verbatim from `crucible_bench.testbench.tasks_code` and
rendered through the model's chat template, so the text matches what the harness
actually sends at eval time — including the role markers and special tokens.

    python scripts/gen_kld_corpus.py -o kld_corpus.txt \
        --chat-template outputs/model-dir --limit 400

Pair with `llama-perplexity --kl-divergence-base` against the f16 GGUF.
"""

from __future__ import annotations

import argparse
from itertools import zip_longest

# (dataset_id, config, split, builder) — builder returns the user message for one
# row, or None to skip it. Kept byte-identical to the harness's own wording: a
# divergence corpus that phrases the task differently is measuring different
# activations than the benchmark will.
SOURCES = [
    (
        "evalplus/humanevalplus", None, "test",
        lambda r: (
            "Complete this Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{r['prompt']}"
        ),
    ),
    (
        "evalplus/mbppplus", None, "test",
        lambda r: (
            "Write a Python function for this task. Return ONLY the code, "
            f"no explanation:\n\n{r['prompt']}\n\n"
            f"The function must satisfy: {(r.get('test_list') or [''])[0]}"
        ),
    ),
    (
        "bigcode/bigcodebench", None, "v0.1.4",
        lambda r: (
            "Complete this Python function. Return ONLY the complete function, "
            f"no explanation:\n\n{r['complete_prompt']}"
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="kld_corpus.txt")
    parser.add_argument("--chat-template", default=None, metavar="MODEL_DIR",
                        help="Render each prompt through this model's chat template. "
                             "Strongly recommended — the harness sends templated "
                             "messages, so an untemplated corpus measures divergence "
                             "on text the model never actually receives.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Cap prompts taken per source (default: all). All three "
                             "sources together are 1682 prompts.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Continue when a source fails. Off by default: a missing "
                             "source silently changes what divergence was measured on.")
    args = parser.parse_args()

    from datasets import load_dataset

    tokenizer = None
    if args.chat_template:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.chat_template, trust_remote_code=True)
        if not getattr(tokenizer, "chat_template", None):
            raise SystemExit(f"{args.chat_template} has no chat_template to apply")

    per_source: list[list[str]] = []
    failures: list[str] = []

    for ds_id, config, split, build in SOURCES:
        print(f"  Loading {ds_id} ({split})...")
        try:
            ds = load_dataset(ds_id, config, split=split)
            rows = ds.select(range(min(args.limit, len(ds)))) if args.limit else ds
            taken = 0
            collected: list[str] = []
            for row in rows:
                message = build(row)
                if not message:
                    continue
                if tokenizer is not None:
                    message = tokenizer.apply_chat_template(
                        [{"role": "user", "content": message}],
                        tokenize=False, add_generation_prompt=True,
                    )
                collected.append(message)
                taken += 1
            per_source.append(collected)
            print(f"    Got {taken}")
        except Exception as e:  # noqa: BLE001 - reported, then decided on below
            print(f"    FAILED: {e}")
            failures.append(f"{ds_id}: {type(e).__name__}: {e}")

    if failures and not args.allow_partial:
        raise SystemExit(
            "divergence corpus is incomplete:\n  " + "\n  ".join(failures)
            + "\n\nA corpus missing a source measures divergence on a different "
              "distribution than the one reported.\nRe-run with --allow-partial only "
              "if you have decided that is acceptable."
        )

    # Round-robin across sources rather than concatenating them. The logits file
    # for a KL-divergence baseline costs ~0.5 MB per token at this vocab size, so
    # the full corpus cannot be run — it has to be truncated with
    # `llama-perplexity --chunks N`. Truncation takes a PREFIX, and a prefix of
    # concatenated sources is one source. Interleaved, any prefix is a mix.
    texts: list[str] = []
    for row in zip_longest(*per_source):
        texts.extend(t for t in row if t is not None)

    with open(args.output, "w") as f:
        for text in texts:
            f.write(text.strip() + "\n")

    chars = sum(len(t) for t in texts)
    print(f"\n  Wrote {len(texts)} prompts ({chars/1e6:.1f}M chars) to {args.output}, "
          f"interleaved across {len(per_source)} sources")
    if tokenizer is not None:
        print("  All rendered through the chat template.")


if __name__ == "__main__":
    main()
