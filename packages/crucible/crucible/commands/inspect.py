"""crucible inspect — report a model's MoE architecture from its config alone."""

from __future__ import annotations

import argparse
import sys

NAME = "inspect"
HELP = "Inspect MoE architecture of a model"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped with the checkpoint. This "
             "runs arbitrary Python from the model repo — only enable it for "
             "sources you trust. Every model in the registry loads without it.",
    )


def run(args) -> None:
    from transformers import AutoConfig

    from crucible.methods.observer import _get_config_value
    from crucible.models.registry import MODEL_REGISTRY, get_model_attrs

    print(f"Loading config for {args.model}...")
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    # Find architecture
    arch = None
    for arch_name in config.architectures or []:
        if arch_name in MODEL_REGISTRY:
            arch = arch_name
            break

    if arch is None:
        print(f"Architectures: {config.architectures}")
        print(f"Supported: {list(MODEL_REGISTRY.keys())}")
        print("Model not in registry — add it to crucible/models/registry.py")
        sys.exit(1)

    attrs = get_model_attrs(arch)
    num_experts = _get_config_value(config, attrs.num_experts_key)
    top_k = _get_config_value(config, attrs.num_experts_per_tok_key)

    print(f"\n{'='*60}")
    print(f"Model:          {args.model}")
    print(f"Architecture:   {arch}")
    print(f"Experts:        {num_experts} total, top-{top_k} active per token")
    print(f"Router path:    layer.{attrs.router}")
    print(f"Experts path:   layer.{attrs.experts}")
    print(f"Expert storage: {attrs.expert_storage}")
    print(f"Fused gate_up:  {attrs.fused_gate_up}")

    # Show compression targets
    print(f"\n{'='*60}")
    print("Compression targets (Q4_K_M estimates):")
    for label, ratio in [("25%", 0.25), ("37.5%", 0.375), ("50%", 0.5)]:
        remaining = int(num_experts * (1 - ratio))
        print(f"  {label} → {remaining} experts")

    print(f"{'='*60}")
