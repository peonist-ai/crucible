"""Quick smoke test: load a model and generate a few responses.

Usage:
    python scripts/smoke_test.py <model_path> [--base-model <hf_id>]

Everything runs under main() — importing this module must not touch the network
or load weights, or `pytest` collecting the repo would try to download a model.
"""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPTS = [
    "Write a Python function that returns the fibonacci sequence up to n terms.",
    "Explain what a mutex is in one paragraph.",
    "Write a binary search function in Python.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate a few responses from a model")
    parser.add_argument("model_path", help="Path to a local model directory or HF id")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Fallback processor source — compressed checkpoints often ship without "
        "a processor config (e.g. google/gemma-4-26B-A4B-it)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    print(f"Loading {args.model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
    except OSError:
        if not args.base_model:
            raise SystemExit(
                f"No processor config in {args.model_path}. Re-run with "
                "--base-model <hf_id> to borrow one from the uncompressed model."
            ) from None
        print(f"  Processor not found locally, loading from {args.base_model}")
        processor = AutoProcessor.from_pretrained(args.base_model)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Loaded.\n")

    for prompt in PROMPTS:
        print(f">>> {prompt}")

        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        response = processor.decode(out[0][input_len:], skip_special_tokens=True)
        print(response[:1500])
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
