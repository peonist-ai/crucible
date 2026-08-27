"""Core types for crucible."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompressionMethod(Enum):
    """Available compression methods."""

    REAP = "reap"
    REAM = "ream"


@dataclass
class ModelAttrs:
    """Maps a model's MoE architecture to common attribute names.

    Each supported model needs one of these to tell crucible where
    the experts, router, and projections live in the model graph.
    Paths are dot-separated, relative to each transformer decoder layer.
    """

    model_class: str
    # Dot-paths relative to each decoder layer
    router: str  # e.g., "router" (Gemma4) or "mlp.gate" (Qwen3)
    experts: str  # e.g., "experts" (Gemma4) or "mlp.experts" (Qwen3)
    # Expert weight attribute names (on experts module for tensor3d, on each expert for modulelist)
    gate_proj: str
    up_proj: str
    down_proj: str
    fused_gate_up: bool
    num_experts_key: str
    num_experts_per_tok_key: str
    # Storage format: "modulelist" (list of expert modules) or "tensor3d" (batched 3D params)
    expert_storage: str = "modulelist"
    # Shared expert (runs on all tokens, not routed). Dot-path relative to decoder layer.
    # Must NOT be pruned. None means no shared expert.
    shared_expert: str | None = None
    # Shared expert gate (sigmoid gate controlling shared expert contribution). None if ungated.
    shared_expert_gate: str | None = None
    # Activation inside the expert FFN, as a torch.nn.functional name ("silu",
    # "gelu", ...). Only consulted for `tensor3d` storage, where there is no
    # per-expert module to call — see observer._resolve_expert_activation, which
    # prefers the module's own act_fn and the config's hidden_act over this.
    # Set it only when a model exposes neither.
    expert_act: str | None = None
    # Group-limited (a.k.a. group-constrained) routing: the router first picks
    # `top_k_group` of `n_group` expert groups, then top_k experts within them.
    # DeepSeek-V3 shaped. These name the *config* keys; None means the family
    # does not use grouped routing and experts are pruned globally per layer.
    n_group_key: str | None = None
    top_k_group_key: str | None = None
    # Additive per-expert router bias used for score correction (DeepSeek's
    # `e_score_correction_bias`). Pruned alongside the router rows when present.
    router_score_bias: str | None = None


@dataclass
class CompressionConfig:
    """Configuration for a compression run."""

    model_id: str
    method: CompressionMethod
    compression_ratio: float = 0.5
    calibration_datasets: list[str] = field(default_factory=list)
    calibration_samples: int = 1024
    max_seq_length: int = 2048
    seed: int = 42
    output_dir: str = "outputs"
    # REAM-specific
    sequential_merging: bool = True
    # Post-compression
    quantize: str | None = None  # e.g. "q4_k_m"


@dataclass
class ExpertScore:
    """Importance score for a single expert in a layer."""

    layer_idx: int
    expert_idx: int
    score: float
    frequency: float = 0.0
    activation_norm: float = 0.0
    router_weight: float = 0.0


@dataclass
class MethodContext:
    """Everything a scorer or a compression method is handed.

    One context for both halves of the pipeline: a scorer receives it with
    `scores` unset and produces them, a method receives it with `scores`
    already resolved (or None, for methods that do their own observation).

    `model`, `tokenizer` and `dataloader` are deliberately untyped — this
    module stays free of torch and transformers imports so the CLI can build
    its argument parser without paying for them.
    """

    model: Any
    tokenizer: Any
    dataloader: Any
    attrs: ModelAttrs
    num_experts: int
    top_k: int
    num_to_keep: int
    ratio: float
    # Parsed CLI options, verbatim. Methods and scorers read the flags they
    # registered themselves rather than growing this dataclass a field per
    # strategy — that coupling is what the registry exists to avoid.
    options: dict[str, Any] = field(default_factory=dict)
    scores: list[list[ExpertScore]] | None = None
    per_layer_keep: list[int] | None = None


@dataclass
class ScoringResult:
    """What a scoring strategy produces."""

    scores: list[list[ExpertScore]]
    # Non-uniform per-layer keep counts, for strategies that plan allocation
    # (pathfinder, adaptive). None means uniform.
    per_layer_keep: list[int] | None = None
    # The ObservationResult, when this strategy ran one. Carries shared-expert
    # stats the caller records alongside the scores.
    observation: Any = None


@dataclass
class MethodResult:
    """What a compression method reports back for the run metadata."""

    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of a compression run."""

    original_params: int
    compressed_params: int
    original_experts_per_layer: int
    remaining_experts_per_layer: int
    method: CompressionMethod
    compression_ratio: float
    output_path: str
