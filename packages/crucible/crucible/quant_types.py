"""What each ggml quantization type costs, and how much it hurts.

The table a budget solver needs: exact size per weight, measured distortion, and
the constraints that make a type inadmissible regardless of how good its numbers
look. Pure data plus arithmetic — no I/O, no llama.cpp, no torch.

Two numbers per type, and they come from different places on purpose.

`bpw` is exact. It is `type_size * 8 / blck_size` read off the block layouts in
`ggml/src/ggml-common.h`, not the round figures in `llama-quantize --help`.
Those are whole-file sizes for a *mixture* — "Q2_K: 2.96G, +3.5199 ppl" describes
the Q2_K **recipe**, which holds attention and embeddings far above 2 bits. The
Q2_K **type** is 2.625 bpw. Confusing the two is how a size budget silently
overruns by 15%.

`rmse` is measured, and is the weaker of the two. It is round-trip
quantize/dequantize error over 1,048,576 synthetic weights (unit Gaussian, 2% of
draws widened 6x to stand in for the outlier structure real weight rows have),
normalised by the input RMS, produced by `ggml_quantize_chunk` itself rather than
by a formula. It is a *relative* ranking, and it carries one systematic bias:

    it was measured with a FLAT imatrix, so it understates the i-quants.

Selecting codebooks against real activation importance is the entire design of
IQ*, so their true standing is better than shown — treat these as a floor for
IQ types and as accurate for K-quants. Replace the column wholesale with
measured per-tensor KL-divergence when a machine is free to produce one; the
solver reads distortion through `Distortion`, precisely so this table can be
swapped without touching the planner.

What the numbers already settle: **IQ3_S and Q3_K are both exactly 3.4375 bpw**,
and IQ3_S measures 6.6% lower error. At identical size that is free, and the
gap only widens with a real imatrix.

Where the bias is visible: IQ2_XXS measures *worse* than IQ1_M (0.5378 against
0.5085) while costing 0.31 more bits. That ordering is an artefact, not a fact
about the format — IQ2_XXS is one of the three types `ggml_quantize_chunk`
refuses to produce without an imatrix, so scoring it with a flat one measures it
at its worst. The ladder is therefore **not monotone**, and nothing may assume
it is. `allocate._hull` drops rungs that cost more for no gain, so an artefact
here makes a type unreachable rather than making a plan wrong; re-measure with a
real imatrix before reading anything into the low IQ rows.
"""

from __future__ import annotations

from dataclasses import dataclass

# Types that `ggml_quantize_chunk` refuses to produce without an importance
# matrix — it asserts rather than degrading, so a plan naming one of these
# without an imatrix aborts the quantize run partway through.
# Source: `ggml_quantize_requires_imatrix` in ggml/src/ggml.c.
REQUIRES_IMATRIX = frozenset({"IQ2_XXS", "IQ2_XS", "IQ1_S"})

# Ternary formats. These exist for models *trained* ternary (BitNet-shaped) and
# encode the same three values either way, which is why TQ1_0 and TQ2_0 measure
# byte-identical distortion despite differing in size. Quantizing ordinary
# weights into them is not a compression trade, it is data loss: both measured
# worse than IQ1_S at more bits. Excluded from planning unless asked for.
TERNARY_ONLY = frozenset({"TQ1_0", "TQ2_0"})


@dataclass(frozen=True)
class QuantType:
    """One ggml tensor encoding."""

    name: str
    block: int          # weights per block
    block_bytes: int    # bytes per block
    rmse: float         # round-trip error / input RMS, flat imatrix (see module docstring)

    @property
    def bpw(self) -> float:
        """Bits per weight. Exact — this is the block layout, not a file size."""
        return self.block_bytes * 8 / self.block

    @property
    def requires_imatrix(self) -> bool:
        return self.name in REQUIRES_IMATRIX

    @property
    def ternary_only(self) -> bool:
        return self.name in TERNARY_ONLY

    def bytes_for(self, n_params: int) -> int:
        """Storage for `n_params` weights, rounded up to whole blocks as ggml stores them."""
        blocks = -(-n_params // self.block)  # ceil
        return blocks * self.block_bytes


def _t(name: str, block: int, block_bytes: int, rmse: float) -> QuantType:
    return QuantType(name=name, block=block, block_bytes=block_bytes, rmse=rmse)


# block/block_bytes: sizeof(block_*) from ggml-common.h, QK_K = 256.
# rmse: measured, see module docstring. F16/BF16/F32 are exact by construction.
QUANT_TYPES: dict[str, QuantType] = {
    q.name: q
    for q in (
        _t("IQ1_S",   256,  50, 0.652378),
        _t("TQ1_0",   256,  54, 0.725387),
        _t("IQ1_M",   256,  56, 0.508505),
        _t("IQ2_XXS", 256,  66, 0.537774),
        _t("TQ2_0",   256,  66, 0.725387),
        _t("IQ2_XS",  256,  74, 0.426451),
        _t("IQ2_S",   256,  82, 0.395476),
        _t("Q2_K",    256,  84, 0.286735),
        _t("IQ3_XXS", 256,  98, 0.240209),
        _t("IQ3_S",   256, 110, 0.191738),
        _t("Q3_K",    256, 110, 0.205299),
        _t("MXFP4",    32,  17, 0.154540),
        _t("IQ4_XS",  256, 136, 0.113077),
        _t("IQ4_NL",   32,  18, 0.111471),
        _t("Q4_0",     32,  18, 0.147608),
        _t("Q4_K",    256, 144, 0.090689),
        _t("Q5_0",     32,  22, 0.074037),
        _t("Q5_K",    256, 176, 0.044846),
        _t("Q6_K",    256, 210, 0.027603),
        _t("Q8_0",     32,  34, 0.009427),
        _t("Q4_1",     32,  20, 0.090005),
        _t("Q5_1",     32,  24, 0.043629),
        _t("BF16",      1,   2, 0.001655),
        _t("F16",       1,   2, 0.000208),
        _t("F32",       1,   4, 0.000000),
    )
}

# The rungs a planner climbs by default: monotonically increasing in both size
# and quality, so a greedy upgrade never pays bytes for a worse encoding. Q4_0
# and Q5_0 are absent deliberately — legacy types a K-quant beats at the very
# same size: Q4_0 measured 0.1476 against Q4_K's 0.0907 at 4.5 bpw, Q5_0 0.0740
# against Q5_K's 0.0448 at 5.5. MXFP4 (0.1545) and IQ4_NL (0.1115) lose to
# IQ4_XS (0.1131) on size, quality, or both. None of them is ever the right rung.
DEFAULT_LADDER: tuple[str, ...] = (
    "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K",
    "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
)


def ladder(
    *,
    have_imatrix: bool = True,
    exclude: frozenset[str] | tuple[str, ...] = (),
    floor: str | None = None,
    ceiling: str | None = None,
) -> tuple[QuantType, ...]:
    """The admissible rungs, ascending by size.

    `exclude` is where backend reality lands: a type can be perfectly good on
    paper and still be the wrong answer on a given machine (Q2_K dequant carries
    a ~7x prefill penalty on gfx1151, so the Strix Halo tier drops it while the
    Metal tier keeps it). Tiers own that list; this module only knows formats.
    """
    excluded = frozenset(exclude)
    out = [
        QUANT_TYPES[n]
        for n in DEFAULT_LADDER
        if n not in excluded and (have_imatrix or n not in REQUIRES_IMATRIX)
    ]
    if floor is not None:
        lo = QUANT_TYPES[floor].bpw
        out = [q for q in out if q.bpw >= lo]
    if ceiling is not None:
        hi = QUANT_TYPES[ceiling].bpw
        out = [q for q in out if q.bpw <= hi]
    if not out:
        raise ValueError(
            "no admissible quantization types left after filtering "
            f"(exclude={sorted(excluded)}, floor={floor}, ceiling={ceiling}, "
            f"have_imatrix={have_imatrix})"
        )
    return tuple(sorted(out, key=lambda q: q.bpw))
