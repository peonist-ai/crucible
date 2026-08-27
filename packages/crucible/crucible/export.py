"""Export compressed models for local inference.

Saves a compressed HuggingFace model as safetensors with updated config,
ready for GGUF conversion via llama.cpp's convert script.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_compressed(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str | Path,
    *,
    compression_metadata: dict | None = None,
    source_model_path: str | Path | None = None,
) -> Path:
    """Save a compressed model and tokenizer to disk.

    Saves in HuggingFace format (safetensors + config.json) so it can be:
      1. Loaded back with AutoModelForCausalLM.from_pretrained()
      2. Converted to GGUF with llama.cpp's convert_hf_to_gguf.py

    Args:
        model: the compressed HuggingFace model.
        tokenizer: the model's tokenizer.
        output_dir: directory to save into.
        compression_metadata: optional dict with compression details
            (method, ratio, calibration, etc.) saved alongside the model.
        source_model_path: path/id of the original uncompressed model. If the
            source has a multimodal wrapper config (architectures ends in
            ForConditionalGeneration with a nested text_config), we rewrap
            the saved text-only config into that same wrapper so vLLM and
            other runtimes that expect the multimodal config type can load it.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights + config (flat text config if loaded via AutoModelForCausalLM)
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer/processor
    tokenizer.save_pretrained(output_dir)

    # For multimodal models, also copy processor configs from source if available
    if hasattr(tokenizer, "image_processor"):
        tokenizer.image_processor.save_pretrained(output_dir)

    # Rewrap config if source was a multimodal wrapper (Qwen3_5Moe, Gemma4 etc.)
    if source_model_path is not None:
        _maybe_rewrap_config(output_dir, source_model_path)

    # Last, so it fixes whichever config shape we ended up writing.
    _strip_absent_mtp_claim(output_dir)

    # Save compression metadata
    if compression_metadata:
        meta_path = output_dir / "compression_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(compression_metadata, f, indent=2)

    return output_dir


def _maybe_rewrap_config(output_dir: Path, source_model_path: str | Path) -> None:
    """If the source model has a multimodal wrapper config, put our saved
    text-only config under its text_config key so loaders that expect the
    wrapper (e.g. vLLM for Qwen3_5Moe) can open it.

    No-op if the source config isn't a wrapper.
    """
    src = Path(source_model_path)
    if not src.exists():
        return
    src_config_path = src / "config.json"
    if not src_config_path.exists():
        return
    src_cfg = json.loads(src_config_path.read_text())
    archs = src_cfg.get("architectures") or []
    if not any(a.endswith("ForConditionalGeneration") for a in archs):
        return
    if "text_config" not in src_cfg:
        return

    saved_cfg_path = output_dir / "config.json"
    saved_cfg = json.loads(saved_cfg_path.read_text())

    # Preserve original text_config model_type, drop transient keys
    new_text_config = dict(saved_cfg)
    for drop in ("architectures", "transformers_version"):
        new_text_config.pop(drop, None)
    new_text_config["model_type"] = src_cfg["text_config"].get(
        "model_type", new_text_config.get("model_type")
    )

    wrapped = dict(src_cfg)
    wrapped["text_config"] = new_text_config
    wrapped["architectures"] = archs  # keep multimodal arch for vLLM

    saved_cfg_path.write_text(json.dumps(wrapped, indent=2))


# Config keys that describe a multi-token-prediction head.
_MTP_CONFIG_KEYS = (
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "num_nextn_predict_layers",
)


def _strip_absent_mtp_claim(output_dir: Path) -> None:
    """Drop MTP config keys when the saved weights contain no MTP tensors.

    `AutoModelForCausalLM` never loads a model's speculative-decoding head, so a
    compressed checkpoint has no `mtp.*` tensors — but the config object it was
    loaded from still carries `mtp_num_hidden_layers`, and `save_pretrained`
    faithfully writes that claim back out. The result is a checkpoint whose
    config advertises a head that is not in the file.

    That is not cosmetic. Current `convert_hf_to_gguf.py` adds the declared MTP
    layers to `block_count`, so the GGUF gets 41 blocks of metadata for 40 blocks
    of tensors, and every loader then fails with
    `check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found`.

    NOTE stripping the key is necessary but not sufficient for that converter:
    with no MTP tensors *and* no declared MTP layers it asserts instead
    (`conversion/qwen.py`, `assert self.opt_num_mtp_layers != 0`). Pass
    `--no-mtp` when converting. The config is corrected here regardless, because
    a config should describe the weights beside it.
    """
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    has_mtp = False
    for shard in output_dir.glob("*.safetensors"):
        from safetensors import safe_open

        with safe_open(shard, framework="pt") as f:
            if any("mtp." in k or "nextn" in k for k in f.keys()):
                has_mtp = True
                break
    if has_mtp:
        return

    config = json.loads(config_path.read_text())
    removed = []
    for section in (config, config.get("text_config")):
        if not isinstance(section, dict):
            continue
        for key in _MTP_CONFIG_KEYS:
            if section.pop(key, None) is not None:
                removed.append(key)
    if removed:
        config_path.write_text(json.dumps(config, indent=2))
        print(
            f"  Dropped MTP config claim ({', '.join(sorted(set(removed)))}) — "
            "no mtp.* tensors in the saved weights.\n"
            "  Convert to GGUF with: convert_hf_to_gguf.py <dir> --no-mtp"
        )


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB from parameter count and dtypes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
