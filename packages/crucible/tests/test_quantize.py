"""Tests for crucible.quantize.

Covers the parts that do not need llm-compressor installed: the full-precision
ignore list (derived from the model registry) and the post-save config fixup.
`build_recipe` and `require_llmcompressor` are thin wrappers over the optional
dependency and are exercised by running the command.
"""

import json
import re

from crucible.models.registry import MODEL_REGISTRY
from crucible.quantize import quant_ignore_patterns, realign_ignore_list


def _held(patterns: list[str], module_name: str) -> bool:
    """Would this module be held at full precision?

    Mirrors how vLLM reads the list: exact string match, regex only for entries
    prefixed `re:`. Asserting against this rather than against the pattern
    spelling means the tests survive a change of escaping style and still catch a
    pattern that stops matching.
    """
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.fullmatch(pattern[3:], module_name):
                return True
        elif pattern == module_name:
            return True
    return False


class TestIgnorePatterns:
    def test_lm_head_always_full_precision(self):
        assert _held(quant_ignore_patterns(), "lm_head")

    def test_routed_experts_are_never_held(self):
        """The routed experts are the entire point of the quantization — no flag
        combination may accidentally exclude them."""
        for attention in (True, False):
            for shared in (True, False):
                patterns = quant_ignore_patterns(
                    keep_attention=attention, keep_shared_expert=shared
                )
                assert not _held(
                    patterns, "model.layers.0.mlp.experts.3.down_proj"
                )
                assert not _held(
                    patterns, "model.layers.0.block_sparse_moe.experts.3.w2"
                )

    def test_every_registry_router_is_held(self):
        """Routers decide which experts fire; quantizing one corrupts routing for
        every token. Their paths differ per family, so the list is derived from
        the registry rather than hardcoded — this asserts that derivation."""
        patterns = quant_ignore_patterns()
        for attrs in MODEL_REGISTRY.values():
            name = f"model.layers.0.{attrs.router}"
            assert _held(patterns, name), f"{attrs.model_class}: {name}"

    def test_shared_expert_gates_are_held(self):
        patterns = quant_ignore_patterns()
        gates = [
            a.shared_expert_gate
            for a in MODEL_REGISTRY.values()
            if a.shared_expert_gate
        ]
        assert gates, "expected at least one registry entry with a gate"
        for gate in gates:
            assert _held(patterns, f"model.layers.0.{gate}")

    def test_routers_survive_every_toggle(self):
        for attention in (True, False):
            for shared in (True, False):
                patterns = quant_ignore_patterns(
                    keep_attention=attention, keep_shared_expert=shared
                )
                for attrs in MODEL_REGISTRY.values():
                    assert _held(patterns, f"model.layers.0.{attrs.router}")

    def test_attention_held_by_default_and_releasable(self):
        held = quant_ignore_patterns()
        assert _held(held, "model.layers.0.self_attn.q_proj")
        assert _held(held, "model.layers.0.linear_attn.out_proj")

        released = quant_ignore_patterns(keep_attention=False)
        assert not _held(released, "model.layers.0.self_attn.q_proj")
        assert not _held(released, "model.layers.0.linear_attn.out_proj")

    def test_shared_expert_held_by_default_and_releasable(self):
        held = quant_ignore_patterns()
        assert _held(held, "model.layers.0.mlp.shared_expert.up_proj")

        released = quant_ignore_patterns(keep_shared_expert=False)
        assert not _held(released, "model.layers.0.mlp.shared_expert.up_proj")
        # The gate is a separate concern and stays regardless.
        assert _held(released, "model.layers.0.mlp.shared_expert_gate")

    def test_patterns_are_sorted_and_unique(self):
        patterns = quant_ignore_patterns()
        assert patterns == sorted(patterns)
        assert len(patterns) == len(set(patterns))


class TestRealignIgnoreList:
    """vLLM matches ignore entries by exact string, so a prefix mismatch makes it
    treat held-back layers as quantized and look for tensors that are not there.
    """

    def _write(self, tmp_path, ignore):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"quantization_config": {"ignore": ignore}}))
        return path

    def test_strips_the_language_model_prefix(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.1.mlp.shared_expert.up_proj",
            ],
        )
        assert realign_ignore_list(path) == 2
        entries = json.loads(path.read_text())["quantization_config"]["ignore"]
        assert entries == [
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.mlp.shared_expert.up_proj",
        ]

    def test_counts_only_changed_entries(self, tmp_path):
        path = self._write(
            tmp_path,
            ["model.language_model.layers.0.self_attn.q_proj", "lm_head"],
        )
        assert realign_ignore_list(path) == 1

    def test_no_op_when_names_already_align(self, tmp_path):
        path = self._write(tmp_path, ["model.layers.0.self_attn.q_proj", "lm_head"])
        before = path.read_text()
        assert realign_ignore_list(path) == 0
        assert path.read_text() == before

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert realign_ignore_list(tmp_path / "nope.json") == 0

    def test_config_without_quantization_section(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"model_type": "qwen3_5_moe"}))
        assert realign_ignore_list(path) == 0

    def test_regex_entries_pass_through_untouched(self, tmp_path):
        path = self._write(tmp_path, [r"re:.*self_attn\..*"])
        assert realign_ignore_list(path) == 0
        entries = json.loads(path.read_text())["quantization_config"]["ignore"]
        assert entries == [r"re:.*self_attn\..*"]
