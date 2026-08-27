"""Reading GGUF files, to the depth a quantization plan needs.

Header, metadata and the tensor directory — never the tensor data. That is
enough to inventory a model (what tensors, what shapes, what encoding) and to
read an importance matrix, and it means planning a 38 GB quantization touches
a few hundred kilobytes at the front of the file.

Dependency-free on purpose. crucible does not depend on llama.cpp at runtime —
`crucible compress` writes safetensors and `convert_hf_to_gguf.py` is somebody
else's script — so taking the `gguf` package as a dependency to count tensors
would be the wrong trade. The container format is small, stable and versioned;
the standard library parses it fine.

Layout, from `ggml/src/gguf.cpp`:

    "GGUF" | version:u32 | n_tensors:u64 | n_kv:u64
    n_kv   x  ( key:str, type:u32, value )
    n_tens x  ( name:str, n_dims:u32, dims:u64[n_dims], ggml_type:u32, offset:u64 )
    padding to general.alignment
    tensor data

A string is u64 length then raw bytes; an array is an element type, a u64 count,
then the elements.
"""

from __future__ import annotations

import array
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# gguf_type, from ggml/include/gguf.h.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)

_SCALARS: dict[int, tuple[str, int]] = {
    _U8: ("<B", 1), _I8: ("<b", 1),
    _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4),
    _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# ggml_type ordinal -> name, from the enum in ggml/include/ggml.h. Only the
# encodings a checkpoint can actually hold are listed; the removed repacking
# variants (Q4_0_4_4 and friends) are gaps on purpose so an old file naming one
# raises instead of being silently mislabelled.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}


@dataclass(frozen=True)
class Tensor:
    """One entry in the tensor directory."""

    name: str
    shape: tuple[int, ...]
    type_name: str
    offset: int

    @property
    def n_params(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def layer(self) -> int | None:
        """The `blk.<N>.` index, or None for a model-level tensor."""
        parts = self.name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "blk" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    @property
    def role(self) -> str:
        """The name with its layer index removed, so tensors group across the stack.

        `blk.7.ffn_down_exps.weight` -> `ffn_down_exps.weight`. Model-level
        tensors are their own role.
        """
        parts = self.name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "blk" and parts[i + 1].isdigit():
                return ".".join(parts[i + 2:])
        return self.name


@dataclass(frozen=True)
class GGUF:
    """A parsed GGUF header."""

    path: Path
    version: int
    kv: dict[str, Any]
    tensors: tuple[Tensor, ...]
    data_offset: int = 0

    @property
    def architecture(self) -> str | None:
        return self.kv.get("general.architecture")

    def by_name(self) -> dict[str, Tensor]:
        return {t.name: t for t in self.tensors}


def _read(f: BinaryIO, fmt: str, size: int) -> Any:
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"truncated GGUF: wanted {size} bytes, got {len(data)}")
    return struct.unpack(fmt, data)[0]


def _read_str(f: BinaryIO) -> str:
    n = _read(f, "<Q", 8)
    # A corrupt length here would otherwise try to allocate petabytes.
    if n > (1 << 28):
        raise ValueError(f"implausible GGUF string length {n} — not a GGUF file?")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return _read(f, fmt, size)
    if vtype == _STR:
        return _read_str(f)
    if vtype == _ARR:
        etype = _read(f, "<I", 4)
        count = _read(f, "<Q", 8)
        return [_read_value(f, etype) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(path: str | Path) -> GGUF:
    """Parse a GGUF's metadata and tensor directory. Does not read tensor data."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file (magic {magic!r})")
        version = _read(f, "<I", 4)
        if version < 2:
            raise ValueError(
                f"{path} is GGUF v{version}; only v2+ has 64-bit counts and is supported"
            )
        n_tensors = _read(f, "<Q", 8)
        n_kv = _read(f, "<Q", 8)

        kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_str(f)
            kv[key] = _read_value(f, _read(f, "<I", 4))

        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read(f, "<I", 4)
            shape = tuple(_read(f, "<Q", 8) for _ in range(n_dims))
            type_id = _read(f, "<I", 4)
            offset = _read(f, "<Q", 8)
            if type_id not in GGML_TYPE_NAMES:
                raise ValueError(f"tensor {name!r} has unknown ggml type {type_id}")
            tensors.append(
                Tensor(name=name, shape=shape, type_name=GGML_TYPE_NAMES[type_id], offset=offset)
            )

        alignment = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data_offset = pos + (-pos % alignment)

    return GGUF(
        path=path, version=version, kv=kv, tensors=tuple(tensors), data_offset=data_offset
    )


def read_f32(gguf: GGUF, tensor: Tensor) -> array.array:
    """Read one F32 tensor's data.

    Only F32, and only for small files — this exists to read importance
    matrices, which are a few megabytes. Model weights are never read here.
    """
    if tensor.type_name != "F32":
        raise ValueError(f"{tensor.name} is {tensor.type_name}, not F32")
    out = array.array("f")
    with gguf.path.open("rb") as f:
        f.seek(gguf.data_offset + tensor.offset)
        out.frombytes(f.read(tensor.n_params * 4))
    if sys.byteorder != "little":
        out.byteswap()
    if len(out) != tensor.n_params:
        raise ValueError(f"{tensor.name}: short read ({len(out)} of {tensor.n_params} floats)")
    return out


def read_imatrix(path: str | Path) -> dict[str, float]:
    """Mean activation energy per tensor, from a llama-imatrix GGUF.

    `llama-imatrix` accumulates the sum of squared activations per input channel
    and the number of rows that contributed, storing them as `<name>.in_sum2`
    and `<name>.counts`. Mean energy per channel is the first divided by the
    second — the same reduction `compute_statistics` does in
    `tools/imatrix/imatrix.cpp`.

    For a MoE 3D expert stack `counts` has one entry per expert, so the file
    carries per-expert importance. That resolution cannot be spent: GGUF stacks
    a layer's experts into one tensor and a tensor has one type. It is averaged
    across the stack here, weighted by how often each expert actually fired, so
    a layer whose experts are rarely routed to is correctly ranked as cheap.

    Returns names as they appear in the model, ready to match against a
    `Tensor.name`. Entries with no data (an expert the calibration never routed
    to) are skipped rather than counted as zero energy.
    """
    imx = read_header(path)
    kind = imx.kv.get("general.type")
    if kind != "imatrix":
        raise ValueError(f"{path} is not an imatrix GGUF (general.type={kind!r})")

    by_name = imx.by_name()
    energies: dict[str, float] = {}
    for tensor in imx.tensors:
        if not tensor.name.endswith(".in_sum2"):
            continue
        base = tensor.name[: -len(".in_sum2")]
        counts_t = by_name.get(f"{base}.counts")
        if counts_t is None:
            continue
        sums = read_f32(imx, tensor)
        counts = read_f32(imx, counts_t)
        n_mat = len(counts)
        if n_mat == 0 or len(sums) % n_mat != 0:
            continue
        row = len(sums) // n_mat
        total, weight = 0.0, 0.0
        for i, c in enumerate(counts):
            if c <= 0:  # expert never routed to by the calibration set
                continue
            total += sum(sums[i * row:(i + 1) * row]) / c
            weight += 1.0
        if weight:
            energies[base] = total / (weight * row)
    return energies
