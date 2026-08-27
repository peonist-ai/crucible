"""Hardware tiers — named presets over the budget flags.

The primitive is a byte budget and a backend. A tier is a thin preset that
expands to those, so nothing here is load-bearing: `--max-bytes 11GiB --backend
metal` reaches exactly the same planner as `--target mac-mini-m4-16gb`, and any
flag given explicitly wins over the preset. The registry exists so the hardware
knowledge lives with crucible rather than being re-derived by every caller.

## Two different objectives wear the same word "best"

A capacity-bound tier has a hard wall: exceed it and the model does not load, so
the goal is maximum quality under a size cap and every unspent byte is waste.

A bandwidth-bound tier has no wall. The Strix Halo holds 104 GiB free and reads
224.2 GiB/s, so what limits it is that decode drags the active weights through
memory once per token. There "best" means the largest model that still clears a
tokens-per-second floor, which points at *more* parameters more aggressively
quantized — the opposite prescription from the 16 GB Mac. Same toolkit, inverted
trade, which is the whole reason a tier carries `bound` rather than just a number.

## What a "weight budget" leaves out

`weight_budget_bytes` counts weights. Three other things share the same memory and
have bitten this fleet:

- **The logits compute buffer, which scales with vocabulary.** At Qwen 3.6's
  248,320-token vocab, `-ub 2048` needs `2048 x 248320 x 4B = 2.03 GiB` — more than
  a fifth of the M4's entire working set, for a buffer that holds no weights.
  Measured 2026-08-25: an 8.78 GiB model plus a 0.31 GiB draft model plus that
  buffer OOMs Metal (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) despite the
  weights fitting with 3 GiB to spare. Dropping to `-ub 512` reclaims 1.5 GiB — but
  `-b` must come down with it, because mismatched `-b`/`-ub` corrupts quantized
  GatedDeltaNet models.
- **A speculative draft model** — see `spec_overhead_bytes`.
- **The KV cache**, which on a GDN hybrid is unusually cheap (only full-attention
  layers cache anything) and is rarely the binding constraint here.

None of this is modelled. A plan that fits `weight_budget_bytes` is necessary, not
sufficient, and on a capacity-bound tier the vocab-scaled buffer deserves checking
before anyone concludes a model "fits".

## Provenance

Budgets are measured where the note says measured and inferred where it says
inferred. `weight_budget` is deliberately below `usable` — a runtime needs room
for the KV cache, the compute graph and its allocator's fragmentation, and a
plan that fills the last byte is a plan that OOMs on a long context.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class Tier:
    """One hardware target."""

    name: str
    description: str
    backend: str                  # metal | cuda | rocm | cpu
    bound: str                    # "capacity" | "bandwidth"
    usable_bytes: int             # what the runtime can actually address
    weight_budget_bytes: int      # of that, what a plan may spend on weights
    exclude_types: tuple[str, ...] = ()
    bandwidth_gibs: float | None = None
    # Bytes a speculative-decoding draft head would take on this tier, if one is
    # wanted. Zero where speculation costs nothing worth budgeting for.
    spec_overhead_bytes: int = 0
    formats: tuple[str, ...] = ("gguf",)
    notes: str = ""

    @property
    def headroom_bytes(self) -> int:
        """What the budget leaves for KV cache, graph and allocator slack."""
        return self.usable_bytes - self.weight_budget_bytes


# Excluded per backend, with a reason. These are properties of the machine, not
# of the format, which is why they live here and not in quant_types.
#
# gfx1151: Q2_K dequant carries a measured ~7x prefill penalty on this GPU while
# being fine on Metal — a quality-per-byte solver would otherwise happily pick it.
_ROCM_GFX1151_EXCLUDE = ("Q2_K",)

TIERS: dict[str, Tier] = {
    "mac-mini-m4-16gb": Tier(
        name="mac-mini-m4-16gb",
        description="Apple M4, 16 GB unified memory",
        backend="metal",
        bound="capacity",
        # Measured on the M4 Mac Mini: recommendedMaxWorkingSetSize is 12.9 GiB.
        usable_bytes=int(12.9 * GIB),
        # ~1.9 GiB held back. This architecture's KV cache is unusually cheap —
        # only the full-attention layers of a GDN hybrid cache anything at all —
        # but the graph and Metal's allocator still need room.
        weight_budget_bytes=int(11.0 * GIB),
        formats=("gguf", "mlx"),
        # ~500 MB for a quantized MTP sidecar. Speculation is not free on a
        # capacity-bound tier: the draft head competes with the weights it is
        # meant to accelerate, so the budget has to name it rather than discover
        # it at load time.
        spec_overhead_bytes=500 * 1000**2,
        notes=(
            "Resolved 2026-08-23: the consumer harness drives tool calls with "
            "native OpenAI tools[] and no GBNF anywhere, so llama.cpp keeps MTP "
            "(the grammar/speculation conflict is specific to GBNF-forced calls). "
            "That leaves MLX needing to beat llama.cpp by more than the 24% MTP is "
            "worth, purely on raw decode, since mlx-lm drops MTP weights at load. "
            "Deprioritised on that basis; GGUF + a grafted MTP sidecar is the path."
        ),
    ),
    "rtx-3090-24gb": Tier(
        name="rtx-3090-24gb",
        description="NVIDIA RTX 3090, 24 GB GDDR6X",
        backend="cuda",
        bound="capacity",
        # Inferred, not measured — no NVIDIA hardware in this fleet. 24 GiB card,
        # ~1 GiB lost to display and context.
        usable_bytes=23 * GIB,
        weight_budget_bytes=21 * GIB,
        formats=("gguf", "exl3"),
        notes=(
            "INFERRED budget — never measured here. The reason this tier is "
            "interesting is EXL3: QTIP-style trellis quantization with Hadamard "
            "incoherence processing is the strongest thing available at 3-4 bpw, "
            "and it is CUDA-only, so it is reachable on this tier and nowhere "
            "else in the fleet."
        ),
    ),
    "strix-halo-128gb": Tier(
        name="strix-halo-128gb",
        description="AMD Radeon 8060S (gfx1151), 128 GB unified memory",
        backend="rocm",
        bound="bandwidth",
        # Measured: llama-bench reports 104,245 MiB free of 126,976 MiB.
        usable_bytes=int(101.8 * GIB),
        # Not a real constraint on this box. Left generous on purpose: a plan
        # here should be driven by a tokens-per-second floor, not by this number.
        weight_budget_bytes=90 * GIB,
        exclude_types=_ROCM_GFX1151_EXCLUDE,
        bandwidth_gibs=224.2,
        formats=("gguf", "compressed-tensors"),
        notes=(
            "Bandwidth-bound, not capacity-bound: 224.2 GiB/s measured. Decode "
            "ceiling is bandwidth / bytes-read-per-token, so on this tier shrinking "
            "the model buys speed rather than fit, and speculative decoding is "
            "close to free because the memory it costs is not scarce — which is "
            "why spec_overhead_bytes is 0 here and non-zero on the 16 GB tier."
        ),
    ),
}


def get(name: str) -> Tier:
    if name not in TIERS:
        raise ValueError(f"unknown tier {name!r}; known: {', '.join(sorted(TIERS))}")
    return TIERS[name]


def parse_size(text: str) -> int:
    """Parse `11GiB` / `8.5G` / `900MB` / a bare byte count.

    Binary and decimal units are distinguished, because at these magnitudes the
    difference between GiB and GB is about 7% — half the headroom on the 16 GB
    tier.
    """
    s = text.strip().replace("_", "")
    units = (
        ("KIB", 1024), ("MIB", 1024**2), ("GIB", 1024**3), ("TIB", 1024**4),
        ("KB", 1000), ("MB", 1000**2), ("GB", 1000**3), ("TB", 1000**4),
        ("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4),
        ("B", 1),
    )
    upper = s.upper()
    for suffix, mult in units:
        if upper.endswith(suffix):
            number = upper[: -len(suffix)].strip()
            if not number:
                break
            return int(float(number) * mult)
    return int(float(s))


def format_size(n: int) -> str:
    return f"{n / GIB:.3f} GiB"
