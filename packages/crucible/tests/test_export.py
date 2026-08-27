"""Tests for saving a compressed model: the config must describe the weights."""

from __future__ import annotations

import pytest  # noqa: F401  (used by the fixtures below)


class TestMTPConfigClaim:
    """A saved config must not advertise a head the weights do not contain.

    AutoModelForCausalLM never loads a speculative-decoding head, so a compressed
    checkpoint has no mtp.* tensors — but the config it was loaded from still says
    mtp_num_hidden_layers, and save_pretrained writes that claim back out. Current
    convert_hf_to_gguf.py adds declared MTP layers to block_count, producing a GGUF
    with 41 blocks of metadata over 40 blocks of tensors that nothing can load.
    """

    @staticmethod
    def _model_dir(tmp_path, config, tensors):
        import torch
        from safetensors.torch import save_file

        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(__import__("json").dumps(config))
        save_file({k: torch.zeros(2, 2) for k in tensors}, str(tmp_path / "model.safetensors"))
        return tmp_path

    def _load(self, d):
        import json as _json

        return _json.loads((d / "config.json").read_text())

    def test_claim_is_dropped_when_no_mtp_tensors(self, tmp_path):
        from crucible.export import _strip_absent_mtp_claim

        d = self._model_dir(
            tmp_path / "m",
            {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1,
             "mtp_use_dedicated_embeddings": False, "num_hidden_layers": 40},
            ["model.layers.0.self_attn.q_proj.weight"],
        )
        _strip_absent_mtp_claim(d)
        cfg = self._load(d)
        assert "mtp_num_hidden_layers" not in cfg
        assert "mtp_use_dedicated_embeddings" not in cfg
        assert cfg["num_hidden_layers"] == 40  # untouched

    def test_claim_is_kept_when_mtp_tensors_are_present(self, tmp_path):
        from crucible.export import _strip_absent_mtp_claim

        d = self._model_dir(
            tmp_path / "m",
            {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1},
            ["model.layers.0.self_attn.q_proj.weight", "model.mtp.fc.weight"],
        )
        _strip_absent_mtp_claim(d)
        assert self._load(d)["mtp_num_hidden_layers"] == 1

    def test_nested_text_config_is_cleaned_too(self, tmp_path):
        """Multimodal rewrap puts the real config under text_config."""
        from crucible.export import _strip_absent_mtp_claim

        d = self._model_dir(
            tmp_path / "m",
            {"architectures": ["Qwen3_5MoeForConditionalGeneration"],
             "text_config": {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1}},
            ["model.layers.0.self_attn.q_proj.weight"],
        )
        _strip_absent_mtp_claim(d)
        assert "mtp_num_hidden_layers" not in self._load(d)["text_config"]

    def test_no_config_is_not_an_error(self, tmp_path):
        from crucible.export import _strip_absent_mtp_claim

        tmp_path.mkdir(parents=True, exist_ok=True)
        _strip_absent_mtp_claim(tmp_path)  # must not raise
