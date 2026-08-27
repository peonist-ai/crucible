"""Standalone router fine-tuning on an already-compressed model."""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crucible.data import build_calibration_dataloader
from crucible.methods.finetune_router import finetune_router
from crucible.models.registry import get_model_attrs

DEFAULT_PATH = "outputs/run-d/gemma-4-26B-A4B-it-ream-task-aware-37pct"
model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("Loaded.")

# Resolve architecture
arch = None
for name in model.config.architectures or []:
    try:
        attrs = get_model_attrs(name)
        arch = name
        break
    except ValueError:
        continue

if arch is None:
    print(f"Unsupported architecture: {model.config.architectures}")
    sys.exit(1)

print(f"Architecture: {arch}")

# Build calibration data
print("Loading calibration data...")
dataloader = build_calibration_dataloader(
    tokenizer, num_samples=256, max_seq_length=512, batch_size=4, seed=42
)
print(f"  {len(dataloader.dataset)} samples, {len(dataloader)} batches")

# Fine-tune
print(f"\nFine-tuning router ({steps} steps)...")
result = finetune_router(model, dataloader, attrs, steps=steps)

print(f"\n  Trainable params: {result['trainable_params']:,}")
print(f"  Loss: {result['initial_loss']:.4f} -> {result['final_loss']:.4f}")

# Save
output_path = model_path.rstrip("/") + "-ft"
print(f"\nSaving to {output_path}...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
print("Done.")
