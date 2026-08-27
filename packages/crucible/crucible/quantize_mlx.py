"""crucible quantize-mlx — the Apple-silicon output contract.

`crucible compress` drops expert count; this drops precision and hands the result
to MLX. Separate from `crucible quantize` on purpose: that command emits
compressed-tensors for vLLM on ROCm, and almost none of its flags mean anything
here — there is no offload folder, no device map, no attention-implementation
choice, because MLX quantization is weight surgery rather than a calibrated
forward pass.

Why MLX at all, when we already ship GGUF: on Apple silicon it is the runtime
with the shorter path to the hardware, and llama.cpp's own MTP sidecar has no
equivalent in `mlx-lm` (see `MTP_NOTE`). Which one wins is an empirical question
per model; this command exists so the comparison can be made at all.

MLX is an optional dependency — crucible does not depend on it:

    uv pip install "mlx-lm>=0.31"

On Apple silicon that pulls the Metal backend automatically. Elsewhere pick a
backend explicitly: `mlx-lm[cpu]` on any Linux box, `mlx-lm[cuda12]` /
`mlx-lm[cuda13]` on NVIDIA. There is no ROCm backend, so on an AMD machine MLX
runs on the CPU — fine for `rtn`, which never runs a forward pass, and
impractical for anything that does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_INSTALL_HINT = (
    "MLX quantization needs mlx-lm, which crucible does not depend on.\n"
    "  Apple silicon:  uv pip install 'mlx-lm>=0.31'\n"
    "  Linux (CPU):    uv pip install 'mlx-lm[cpu]>=0.31'\n"
    "  Linux (NVIDIA): uv pip install 'mlx-lm[cuda13]>=0.31'\n"
    "There is no ROCm backend — on AMD, MLX runs on the CPU."
)

MTP_NOTE = (
    "mlx-lm drops MTP weights at load (`models/qwen3_5.py`: "
    'weights = {k: v for k, v in weights.items() if "mtp." not in k}), so a '
    "quantized model produced here cannot self-speculate under mlx-lm. The "
    "head is usable today only via mlx-vlm, which loads a separate "
    "`qwen3_5_mtp` checkpoint as --draft-model; native support is still open "
    "upstream as mlx-lm#990."
)

# Methods that need a forward pass over calibration data, and are therefore
# gated on having a machine that can run one. Kept as data so the command can
# warn without re-deriving the list.
CALIBRATED_METHODS = ("gptq",)

QUANT_METHODS = ("rtn", "gptq")

# mlx-lm's own recipes, which reproduce llama.cpp's Q4_K_M-style heuristic
# (more bits in the first and last eighth of the stack, and on down_proj/v_proj).
# Passed through rather than reimplemented.
MLX_RECIPES = ("mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6")

# Substrings identifying the modules our validated GGUF scheme holds at high
# precision. Matched against the module path mlx-lm passes to the predicate,
# e.g. `language_model.model.layers.7.self_attn.q_proj`.
_ATTENTION_HINTS = (".self_attn.",)
_EMBEDDING_HINTS = ("embed_tokens", "lm_head")


@dataclass(frozen=True)
class MLXModules:
    """The mlx-lm surface we use, resolved once so callers never import it."""

    mx: Any
    nn: Any
    load: Any
    save: Any
    quantize_model: Any
    compute_bits_per_weight: Any


def require_mlx() -> MLXModules:
    """Import mlx-lm, or fail with an install hint naming the right backend."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.utils import (
            compute_bits_per_weight,
            load,
            quantize_model,
            save,
        )
    except ImportError as exc:  # pragma: no cover - exercised by running it
        raise SystemExit(f"{_INSTALL_HINT}\n\n(import failed: {exc})") from exc

    return MLXModules(
        mx=mx,
        nn=nn,
        load=load,
        save=save,
        quantize_model=quantize_model,
        compute_bits_per_weight=compute_bits_per_weight,
    )


def load_for_quantization(
    mlx: MLXModules, model_path: str, *, trust_remote_code: bool
) -> tuple[Any, Any, dict]:
    """`mlx_lm.utils.load`, tolerating the kwarg it grew between releases.

    0.31.3 takes `trust_remote_code` only nested inside `tokenizer_config`; main
    also accepts it as a top-level argument. Passing the top-level form to 0.31.3
    is a TypeError, so ask the signature rather than pinning a version.

    `lazy=True` matters on a large checkpoint: weights stay unmaterialised until
    each module is quantized, so peak memory tracks the largest expert stack
    rather than the whole model.
    """
    import inspect

    kwargs: dict[str, Any] = {
        "lazy": True,
        "return_config": True,
        "tokenizer_config": {"trust_remote_code": trust_remote_code},
    }
    if "trust_remote_code" in inspect.signature(mlx.load).parameters:
        kwargs["trust_remote_code"] = trust_remote_code

    return mlx.load(model_path, **kwargs)


def crucible_quant_predicate(
    *,
    bits: int,
    group_size: int,
    high_bits: int,
    high_group_size: int,
    keep_attention_high: bool = True,
    keep_embeddings_high: bool = True,
) -> Callable[[str, Any], bool | dict]:
    """The scheme we validated in GGUF, expressed for MLX.

    The shipped Q3K-mixed build holds token embeddings, the output tensor and
    every attention projection at Q8_0 while the experts take Q3_K, because in a
    35B-A3B nearly all the weight is expert weight: dropping attention with it
    buys almost no size and costs disproportionately. Same reasoning, same split.

    Returns `True` (meaning "quantize at the command's defaults") for everything
    else, so experts, the shared expert and the GDN projections take `bits`.

    Note what this deliberately does NOT decide: the routers. Those are the
    model's business — see `compose_with_model_predicate`.
    """
    high = {"group_size": high_group_size, "bits": high_bits}

    def predicate(path: str, module: Any) -> bool | dict:
        if keep_attention_high and any(hint in path for hint in _ATTENTION_HINTS):
            return high
        if keep_embeddings_high and any(hint in path for hint in _EMBEDDING_HINTS):
            return high
        return True

    return predicate


def compose_with_model_predicate(
    model: Any, predicate: Callable[[str, Any], bool | dict] | None
) -> Callable[[str, Any], bool | dict] | None:
    """Let the model's own predicate win where it has an opinion.

    `mlx_lm.utils.quantize_model` resolves the predicate as

        quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)

    — a plain `or`. So passing any predicate of our own REPLACES the model's
    rather than refining it, and for Qwen 3.5/3.6 MoE the model's predicate is
    the only thing holding `mlp.gate` and `shared_expert_gate` at 8 bits
    (`models/qwen3_5.py`). Quantizing a router to 4 bits perturbs the argmax over
    experts, which is a different and worse failure than a slightly noisy expert:
    it changes *which* expert runs. We already hit this once on the AWQ path.

    The composition rule is "a dict is an explicit opinion, `True` is a default":
    if the model's predicate returns a dict for a module it has asked for
    specific treatment, so honour it. Otherwise ours decides.
    """
    model_predicate = getattr(model, "quant_predicate", None)
    if model_predicate is None:
        return predicate
    if predicate is None:
        return model_predicate

    def composed(path: str, module: Any) -> bool | dict:
        theirs = model_predicate(path, module)
        if isinstance(theirs, dict):
            return theirs
        if theirs is False:
            return False
        return predicate(path, module)

    return composed


def resolve_predicate(
    model: Any,
    *,
    recipe: str | None,
    bits: int,
    group_size: int,
    high_bits: int,
    high_group_size: int,
    keep_attention_high: bool,
    keep_embeddings_high: bool,
) -> Callable[[str, Any], bool | dict] | None:
    """Pick the per-module bit-allocation policy and compose it with the model's.

    `recipe` selects one of mlx-lm's own mixed schemes and takes precedence; with
    no recipe we use crucible's attention-high split unless both `keep_*` flags
    are off, in which case there is nothing to say and the model's own predicate
    is left to act alone.
    """
    if recipe is not None:
        from mlx_lm.convert import mixed_quant_predicate_builder

        predicate = mixed_quant_predicate_builder(recipe, model, group_size)
    elif keep_attention_high or keep_embeddings_high:
        predicate = crucible_quant_predicate(
            bits=bits,
            group_size=group_size,
            high_bits=high_bits,
            high_group_size=high_group_size,
            keep_attention_high=keep_attention_high,
            keep_embeddings_high=keep_embeddings_high,
        )
    else:
        predicate = None

    return compose_with_model_predicate(model, predicate)


def estimate_size_gb(num_params: int, bits_per_weight: float) -> float:
    """Bytes on disk for `num_params` weights at a measured BPW."""
    return num_params * bits_per_weight / 8 / 2**30


def resolve_module_bits(
    path: str,
    module: Any,
    predicate: Callable[[str, Any], bool | dict] | None,
    *,
    bits: int,
    group_size: int,
) -> tuple[int, int] | None:
    """What width will this module actually get? `None` means "left alone".

    Mirrors the `wrapped_predicate` closure inside `mlx_lm.utils.quantize_model`,
    including the divisibility guard that skips a module whose last dim is not a
    multiple of `group_size` — that guard is silent at runtime, so a module can
    end up at full precision without anything being logged.
    """
    if not hasattr(module, "to_quantized"):
        return None
    if module.weight.shape[-1] % group_size != 0:
        return None
    decision: bool | dict = True
    if predicate is not None:
        decision = predicate(path, module)
    if decision is False:
        return None
    if isinstance(decision, dict):
        return decision.get("bits", bits), decision.get("group_size", group_size)
    return bits, group_size


def plan_quantization(
    model: Any,
    predicate: Callable[[str, Any], bool | dict] | None,
    *,
    bits: int,
    group_size: int,
) -> dict:
    """Project the bit allocation without quantizing anything.

    A real run on a CPU-only MLX build is an hour or more, and the two ways it
    silently disappoints — a module skipped by the divisibility guard, a
    predicate that does not match the paths it was written for — are both visible
    here in seconds.

    The projection charges each quantized weight `bits + 32 / group_size`: the
    packed weight plus one bf16 scale and one bf16 bias per group. That is what
    makes `--group-size 32` at 4 bits cost 5.0 bits/weight rather than 4.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    total_params = sum(
        array.size for _, array in tree_flatten(model.parameters())
    )

    by_bits: dict[int, dict[str, int]] = {}
    quantized_params = 0
    quantized_bits = 0.0
    skipped_modules = 0
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        resolved = resolve_module_bits(
            path, module, predicate, bits=bits, group_size=group_size
        )
        if resolved is None:
            skipped_modules += 1
            continue
        module_bits, module_group = resolved
        size = module.weight.size
        entry = by_bits.setdefault(module_bits, {"modules": 0, "params": 0})
        entry["modules"] += 1
        entry["params"] += size
        quantized_params += size
        quantized_bits += size * (module_bits + 32 / module_group)

    unquantized_params = total_params - quantized_params
    total_bits = quantized_bits + unquantized_params * 16
    return {
        "total_params": total_params,
        "quantized_params": quantized_params,
        "unquantized_params": unquantized_params,
        "unquantized_modules": skipped_modules,
        "by_bits": by_bits,
        "bits_per_weight": total_bits / max(1, total_params),
    }


@dataclass(frozen=True)
class CalibrationTokens:
    """A rectangular calibration corpus for MLX, plus how it was built."""

    tokens: Any  # mx.array, [windows, sequence_length]
    stats: dict


def build_calibration_tokens(
    tokenizer: Any,
    *,
    profile: str,
    num_samples: int,
    sequence_length: int,
    seed: int,
) -> CalibrationTokens:
    """Our calibration mix, shaped the way MLX's quantizers want it.

    This is the substitution that makes the command worth having. Every
    calibrated quantizer in mlx-lm (`quant/awq.py`, `quant/gptq.py`,
    `quant/dynamic_quant.py`) calls `mlx_lm.quant.utils.load_data`, which
    downloads one fixed file of generic English web text and tokenizes it. For a
    coding/tool-use specialist that is the wrong distribution to be measuring
    importance on — and calibration mix is the biggest lever we have, larger than
    the compression ratio. So: same `build_calibration_texts()` the expert scorer
    and the imatrix pass use, rendered through the same chat template.

    Two deliberate differences from `quantize.build_calibration_windows`:

    - Only FULL windows are kept. MLX quantizers slice `data[s:s+batch_size]` and
      run the batch as one array, so the corpus has to be rectangular; a ragged
      tail cannot be padded without calibrating on pad tokens. At the default
      sequence length a median conversation yields dozens of full windows, so the
      dropped tails are a rounding error rather than the systematic
      end-of-conversation bias that truncation would introduce.
    - `num_samples` counts windows, as it does there — it is what drives runtime.
    """
    import random

    from crucible.data import build_calibration_texts

    mx = require_mlx().mx

    # mlx-lm hands back a TokenizerWrapper; the mix builder wants the real HF
    # tokenizer, and setting pad_token through the proxy is not reliable.
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)

    texts = build_calibration_texts(
        hf_tokenizer, profile=profile, num_samples=num_samples, seed=seed
    )

    rng = random.Random(seed)
    windows: list[list[int]] = []
    raw_lengths: list[int] = []
    for text in texts:
        ids = hf_tokenizer(text, add_special_tokens=False).input_ids
        raw_lengths.append(len(ids))
        for start in range(0, len(ids) - sequence_length + 1, sequence_length):
            windows.append(ids[start:start + sequence_length])

    rng.shuffle(windows)
    selected = windows[:num_samples]
    if not selected:
        raise ValueError(
            "Calibration produced no usable windows — every conversation was "
            f"shorter than the {sequence_length}-token window. Lower "
            "--sequence-length or raise --samples."
        )

    lengths = sorted(raw_lengths)
    stats = {
        "profile": profile,
        "seed": seed,
        "sequence_length": sequence_length,
        "source_conversations": len(texts),
        "conv_token_median": lengths[len(lengths) // 2],
        "conv_token_p95": lengths[int(len(lengths) * 0.95)],
        "windows_available": len(windows),
        "windows_requested": num_samples,
        "windows_used": len(selected),
        "total_calibration_tokens": len(selected) * sequence_length,
    }
    return CalibrationTokens(tokens=mx.array(selected), stats=stats)


def gptq_router_modules(model: Any) -> list[str]:
    """Paths GPTQ would quantize that the model's predicate protects.

    `mlx_lm.quant.gptq.gptq_quantize` walks every `nn.Linear` and `SwitchLinear`
    and quantizes all of them at the same width. It never consults
    `model.quant_predicate`, so on a MoE it takes the routers down to `bits`
    along with the experts — the exact failure `compose_with_model_predicate`
    exists to prevent on the `rtn` path. Reported so the command can say so out
    loud rather than silently shipping a 4-bit router.
    """
    predicate = getattr(model, "quant_predicate", None)
    if predicate is None:
        return []

    import mlx.nn as nn
    from mlx.utils import tree_flatten

    protected = []
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        if isinstance(predicate(path, module), dict):
            protected.append(path)
    return protected
