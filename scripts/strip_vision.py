"""Strip vision tower from a multimodal model before GGUF conversion."""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) < 2:
    raise SystemExit("usage: strip_vision.py <model_path> [output_path]")

model_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else model_path.rstrip("/") + "-text-only"

print(f"Loading {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Count params before
total_before = sum(p.numel() for p in model.parameters())

# Find and remove vision components
removed = []
for name in list(model._modules.keys()):
    if "vision" in name.lower() or "embed_vision" in name.lower():
        removed.append(name)

# Remove from the inner model too
inner = getattr(model, "model", model)
for name in list(inner._modules.keys()):
    if "vision" in name.lower() or "embed_vision" in name.lower():
        delattr(inner, name)
        removed.append(f"model.{name}")

total_after = sum(p.numel() for p in model.parameters())
stripped = total_before - total_after

print(f"Removed: {removed}")
print(f"Params: {total_before/1e9:.2f}B -> {total_after/1e9:.2f}B ({stripped/1e6:.0f}M stripped)")

print(f"Saving to {output_path}...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
print("Done.")
