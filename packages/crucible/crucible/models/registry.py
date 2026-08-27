"""Model adapter registry — maps model architectures to their MoE internals."""

from crucible.types import ModelAttrs

# Each entry tells crucible where to find experts, routers, and projections
# in a given model architecture. Add new models here.
#
# None of the families below use group-limited routing, so none set
# `n_group_key` / `top_k_group_key`. Checked against the published configs
# (2026-08-17): Qwen3.5-MoE carries `num_experts` and `num_experts_per_tok` and
# no `n_group` or `topk_group` at all, and REAP at 48% came out near-lossless on
# it — which it could not have if a grouping constraint were being violated
# silently. A DeepSeek-V3-shaped model does need those keys; set them and REAP
# partitions the budget across groups instead of ranking globally.
MODEL_REGISTRY: dict[str, ModelAttrs] = {
    # Gemma 4 26B MoE — 128 experts, 8 active per token
    # Router and experts are siblings at the layer level (not under mlp)
    # Uses 3D parameter tensors: gate_up_proj [num_experts, 2*intermediate, hidden]
    "Gemma4ForConditionalGeneration": ModelAttrs(
        model_class="Gemma4ForConditionalGeneration",
        router="router",
        experts="experts",
        gate_proj="gate_up_proj",
        up_proj="gate_up_proj",
        down_proj="down_proj",
        fused_gate_up=True,
        num_experts_key="num_experts",
        num_experts_per_tok_key="top_k_experts",
        expert_storage="tensor3d",
    ),
    # Qwen3 MoE — 128 experts (30B-A3B variant)
    "Qwen3MoeForCausalLM": ModelAttrs(
        model_class="Qwen3MoeForCausalLM",
        router="mlp.gate",
        experts="mlp.experts",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
        fused_gate_up=False,
        num_experts_key="num_local_experts",
        num_experts_per_tok_key="num_experts_per_tok",
        expert_storage="modulelist",
    ),
    # Qwen 3.5/3.6 MoE — 256 experts + 1 shared, 8 routed per token
    # Architecture: Qwen3_5MoeForConditionalGeneration (multimodal wrapper)
    # Router: simple softmax + top-k + renormalization (no per-expert scale)
    # Expert storage: tensor3d with fused gate_up_proj [num_experts, 2*intermediate, hidden]
    # Shared expert: separate MLP that runs on ALL tokens, gated by sigmoid
    # Layer types alternate: 3x linear_attention + 1x full_attention
    "Qwen3_5MoeForConditionalGeneration": ModelAttrs(
        model_class="Qwen3_5MoeForConditionalGeneration",
        router="mlp.gate",
        experts="mlp.experts",
        gate_proj="gate_up_proj",
        up_proj="gate_up_proj",
        down_proj="down_proj",
        fused_gate_up=True,
        num_experts_key="num_experts",
        num_experts_per_tok_key="num_experts_per_tok",
        expert_storage="tensor3d",
        shared_expert="mlp.shared_expert",
        shared_expert_gate="mlp.shared_expert_gate",
    ),
    # Also register the text-only variant (causal LM without vision)
    "Qwen3_5MoeForCausalLM": ModelAttrs(
        model_class="Qwen3_5MoeForCausalLM",
        router="mlp.gate",
        experts="mlp.experts",
        gate_proj="gate_up_proj",
        up_proj="gate_up_proj",
        down_proj="down_proj",
        fused_gate_up=True,
        num_experts_key="num_experts",
        num_experts_per_tok_key="num_experts_per_tok",
        expert_storage="tensor3d",
        shared_expert="mlp.shared_expert",
        shared_expert_gate="mlp.shared_expert_gate",
    ),
    # Mixtral 8x7B
    "MixtralForCausalLM": ModelAttrs(
        model_class="MixtralForCausalLM",
        router="block_sparse_moe.gate",
        experts="block_sparse_moe.experts",
        gate_proj="w1",
        up_proj="w3",
        down_proj="w2",
        fused_gate_up=False,
        num_experts_key="num_local_experts",
        num_experts_per_tok_key="num_experts_per_tok",
        expert_storage="modulelist",
    ),
}


def get_model_attrs(model_class_name: str) -> ModelAttrs:
    """Look up MoE attributes for a model class.

    The failure here is the most likely first thing a new user hits, so it says
    what to do about it rather than only what went wrong. Adding an architecture
    is one entry in this file — no CLI change, no method change.
    """
    if model_class_name not in MODEL_REGISTRY:
        supported = "\n".join(f"  - {name}" for name in MODEL_REGISTRY)
        raise ValueError(
            f"Unsupported model architecture: {model_class_name}\n\n"
            f"Supported:\n{supported}\n\n"
            f"Adding one is a single ModelAttrs entry in "
            f"crucible/models/registry.py declaring where this architecture "
            f"keeps its router, experts and projections, and whether the "
            f"experts are a ModuleList or a stacked 3D tensor. Nothing else "
            f"needs to change. See CONTRIBUTING.md.\n\n"
            f"If the architecture name looks wrong, note that loading a "
            f"*ForConditionalGeneration checkpoint through AutoModelForCausalLM "
            f"unwraps it and reports the inner class instead."
        )
    return MODEL_REGISTRY[model_class_name]
