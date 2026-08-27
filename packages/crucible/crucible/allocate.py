"""Spend a byte budget across a model's tensors.

The piece Unsloth's Dynamic GGUFs have that a hand-written `--tensor-type` list
does not: instead of one type for the whole file plus a few overrides someone
guessed at, every tensor group is assigned the encoding that buys the most
accuracy per byte until the budget runs out.

The problem is a multiple-choice knapsack — each group picks exactly one rung
off the quality ladder, and the picks share one size cap. It is solved here by
marginal gain: start every group at its cheapest admissible rung, then keep
applying whichever single upgrade has the best error-reduction-per-byte, until
nothing affordable is left. Pruning each group's rungs to their lower convex
hull first makes that greedy exact for the continuous relaxation, and the
integrality gap on a model with hundreds of groups is a rounding error.

## The objective

Quantizing `W` to type `t` perturbs the layer output by `ΔW·x`. Treating the
per-weight error as independent of the activations,

    E[||ΔW x||²]  =  Σ_ij ΔW_ij² E[x_j²]  ≈  rmse(t)² σ_W² · Σ_i Σ_j E[x_j²]

which factors into a term that depends only on the encoding, `rmse(t)²`, and a
term that depends only on the tensor, `n_params × mean_j E[x_j²]`. The second
factor is exactly what an importance matrix measures — llama-imatrix accumulates
`Σ x_j²` per input channel and the count to divide by. So:

    error(group, type) = rmse(type)² × n_params(group) × activation_energy(group)

`σ_W²` is dropped: separating it needs the weights themselves. **That makes raw
energy meaningless ACROSS roles**, and the failure is not subtle — unfloored, the
solver assigns `ssm_out` and `attn_output` **1.75 bpw** and destroys the model.

The reason is worth stating exactly, because the obvious fix is not the fix.
Measured on Qwen 3.6 REAP-48 at layer 20:

    E[x^2]  0.9218   attn_qkv, attn_gate, ssm_alpha    <- read the input-normed hidden
    E[x^2]  0.8927   ffn_gate_exps, ffn_gate_shexp     <- read the post-attn-normed hidden
    E[x^2]  0.0110   ffn_down_exps, ffn_down_shexp     <- read the FFN intermediate
    E[x^2]  2.78e-4  ssm_out                           <- reads the delta-net output

**Energy clusters by position relative to the nearest RMSNorm, not by
importance.** Anything reading a normed hidden state measures ~0.9 by
construction. σ_W does not rescue this: measured, weight RMS spans only 2.4x
(0.0086-0.021) across these same tensors, and including σ_W² *widens* the spread
from 3,317x to 4,811x rather than closing it. The signal is simply not in the
imatrix.

So energy is used only *within* a role — see `normalize_energy_per_role`, which
divides by the role's parameter-weighted mean and answers "which layers of this
role run hotter than their siblings", a comparison where the graph-position
artefact cancels. Allocation *between* roles needs a different measurement
entirely: `Group.sensitivity`, calibrated from measured per-role KL-divergence
(`scripts/measure_role_sensitivity.py`). Structural floors are the stand-in until
that table exists.

With no imatrix every group's energy is 1 and the objective degenerates to
`n_params × rmse²`, which spends bits where the parameters are. That is a
reasonable floor and a bad ceiling: it is blind to the fact that a 126M-parameter
shared expert can carry more of a forward pass than 16.7B parameters of routed
experts. Importance is what tells them apart, so plan with an imatrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from crucible.quant_types import QuantType


@dataclass(frozen=True)
class Group:
    """Tensors planned together, and therefore assigned one encoding.

    Granularity is the caller's choice. Grouping by role gives a compact,
    legible plan; grouping by individual tensor lets the budget vary layer by
    layer, which is where most of the headroom is once the obvious role-level
    mistakes are fixed.

    Note what cannot be a group: a single expert. GGUF stacks every expert of a
    layer into one 3D tensor (`ffn_down_exps.weight` is `[512, 2048, 133]`), and
    a tensor has exactly one type. Per-expert bit allocation is unreachable in
    this format no matter how good the per-expert saliency data is — the finest
    real granularity is per layer.
    """

    key: str
    pattern: str        # anchored regex, for --tensor-type-file
    n_params: int       # TOTAL across the group, not per tensor
    role: str | None = None     # what this group is, independent of granularity
    n_tensors: int = 1
    energy: float = 1.0     # mean activation energy from the imatrix; 1.0 = unknown
    # Measured end-to-end cost of degrading this role, relative to the mean role.
    # 1.0 = unmeasured. This is the term raw energy cannot supply: energy says how
    # large a tensor's inputs are, which clusters by position relative to the
    # nearest RMSNorm, not by how much the output distribution moves.
    sensitivity: float = 1.0
    pinned: str | None = None   # force this type, exempt from the search
    quantizable: bool = True    # False: llama.cpp keeps it F32 whatever we say

    @property
    def importance(self) -> float:
        return self.n_params * self.energy * self.sensitivity

    def bytes_for(self, quant: QuantType) -> int:
        """Storage for the whole group under `quant`.

        Blocks are rounded up per *tensor*, not once over the total, because
        that is how ggml lays them out. Assumes the tensors in a group are the
        same shape — true of a role across layers, and where it is not the error
        is bounded by one block per tensor.
        """
        per = self.n_params // self.n_tensors
        rem = self.n_params - per * self.n_tensors
        total = quant.bytes_for(per) * self.n_tensors
        return total + (quant.bytes_for(rem) if rem else 0)


@dataclass(frozen=True)
class Choice:
    """One group's assigned encoding."""

    group: Group
    quant: QuantType
    n_bytes: int

    @property
    def error(self) -> float:
        return self.quant.rmse**2 * self.group.importance


@dataclass(frozen=True)
class Plan:
    """A complete assignment and what it costs."""

    choices: tuple[Choice, ...]
    budget_bytes: int
    fixed_bytes: int      # groups llama.cpp will not quantize

    @property
    def quantized_bytes(self) -> int:
        return sum(c.n_bytes for c in self.choices)

    @property
    def total_bytes(self) -> int:
        return self.quantized_bytes + self.fixed_bytes

    @property
    def total_params(self) -> int:
        return sum(c.group.n_params for c in self.choices)

    @property
    def bpw(self) -> float:
        """Effective bits per weight over the quantized tensors only."""
        return self.quantized_bytes * 8 / self.total_params if self.total_params else 0.0

    @property
    def error(self) -> float:
        return sum(c.error for c in self.choices)

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.budget_bytes


def _hull(rungs: tuple[QuantType, ...], group: Group) -> list[QuantType]:
    """Drop rungs no budget would ever pick.

    A rung is dominated when another is at least as small *and* at least as
    accurate — Q4_0 against Q4_K at an identical 4.5 bpw. What survives is the
    lower convex hull of (bytes, error): the sequence whose marginal cost per
    unit of error reduction only ever gets worse, which is what makes a greedy
    pass over marginal gains optimal rather than merely plausible.
    """
    pts = sorted(
        ((group.bytes_for(q), q.rmse**2 * group.importance, q) for q in rungs),
        key=lambda p: (p[0], p[1]),
    )
    # Strictly decreasing error as size grows.
    monotone: list[tuple[int, float, QuantType]] = []
    for b, e, q in pts:
        while monotone and monotone[-1][1] <= e:
            if monotone[-1][0] == b:
                monotone.pop()
            else:
                break
        if not monotone or e < monotone[-1][1]:
            monotone.append((b, e, q))

    # Convex hull: discard any rung whose marginal gain beats the one before it,
    # since the pair would then be taken together or not at all.
    hull: list[tuple[int, float, QuantType]] = []
    for pt in monotone:
        while len(hull) >= 2:
            (b0, e0, _), (b1, e1, _) = hull[-2], hull[-1]
            b2, e2, _ = pt
            # slope from b0 and slope from b1, both negative; drop b1 if it is
            # not steep enough to be worth stopping at.
            if (e1 - e0) * (b2 - b1) >= (e2 - e1) * (b1 - b0):
                hull.pop()
            else:
                break
        hull.append(pt)
    return [q for _, _, q in hull]


def plan(
    groups: list[Group],
    ladder: tuple[QuantType, ...],
    budget_bytes: int,
    *,
    pins: dict[str, str] | None = None,
    floors: dict[str, str] | None = None,
) -> Plan:
    """Assign each group an encoding, maximising accuracy inside `budget_bytes`.

    `pins` forces groups to a named type, overriding `Group.pinned`. A key is
    matched against `Group.key` first and `Group.role` second, so the same
    `--pin attn_gate.weight=Q6_K` reaches one role-level group or all forty
    per-layer ones without the caller restating it per layer.

    A pin matching nothing raises. Silently ignoring it would let a plan look
    like it honoured an instruction it never applied — and in per-layer mode,
    where keys are full tensor names, every role-keyed pin would quietly vanish.

    A pin is honoured even when it does not fit: the Plan reports `fits == False`
    rather than substituting something cheaper, because a budget quietly missed
    by 400 MB is the failure that only shows up when the model will not load.

    `floors` sets a *minimum* encoding per key or role, leaving the search free to
    spend more. That is where cross-role knowledge belongs: per-role energies are
    not comparable without σ_W, so something has to stop the solver from handing
    `attn_output` 1.75 bpw because its inputs happen to run at a small scale.
    Unlike a pin, a floor does not stop the search from going higher when the
    measurement says a layer deserves it.
    """
    from crucible.quant_types import QUANT_TYPES

    pins = pins or {}
    floors = floors or {}
    unused_floors = set(floors)
    for name in floors.values():
        if name not in QUANT_TYPES:
            raise ValueError(f"unknown quantization type in floors: {name!r}")
    if not ladder:
        raise ValueError("empty quantization ladder — nothing to assign")

    # llama.cpp leaves these F32 regardless of what a plan asks for.
    fixed = sum(g.bytes_for(QUANT_TYPES["F32"]) for g in groups if not g.quantizable)
    movable = [g for g in groups if g.quantizable]

    unused_pins = set(pins)
    chosen: dict[str, QuantType] = {}
    options: dict[str, list[QuantType]] = {}
    for g in movable:
        # Key before role, so `blk.0.x=Q8_0` overrides a blanket `x=Q6_K`. Both
        # count as used: the role pin was applicable, merely outranked, and
        # reporting it unmatched would be a false alarm.
        pinned = g.pinned
        applicable = [c for c in (g.key, g.role) if c is not None and c in pins]
        if applicable:
            pinned = pins[applicable[0]]
            unused_pins.difference_update(applicable)
        if pinned is not None:
            if pinned not in QUANT_TYPES:
                raise ValueError(f"{g.key}: unknown quantization type {pinned!r}")
            chosen[g.key] = QUANT_TYPES[pinned]
            options[g.key] = []
            continue
        allowed = ladder
        for candidate in (g.key, g.role):
            if candidate is not None and candidate in floors:
                floor_bpw = QUANT_TYPES[floors[candidate]].bpw
                allowed = tuple(q for q in ladder if q.bpw >= floor_bpw)
                unused_floors.discard(candidate)
                break
        if not allowed:
            raise ValueError(
                f"{g.key}: floor {floors.get(g.key) or floors.get(g.role)!r} is above "
                "every rung on the ladder"
            )
        rungs = _hull(allowed, g)
        options[g.key] = rungs
        chosen[g.key] = rungs[0]

    if unused_floors:
        known = sorted({g.role or g.key for g in movable})
        raise ValueError(
            f"floor(s) matched no tensor: {', '.join(sorted(unused_floors))}\n"
            f"known roles: {', '.join(known)}"
        )

    if unused_pins:
        known = sorted({g.role or g.key for g in movable})
        raise ValueError(
            f"pin(s) matched no tensor: {', '.join(sorted(unused_pins))}\n"
            f"known roles: {', '.join(known)}"
        )

    by_key = {g.key: g for g in movable}

    def cost(key: str, q: QuantType) -> int:
        return by_key[key].bytes_for(q)

    spent = fixed + sum(cost(k, q) for k, q in chosen.items())

    # Greedy on marginal gain. Each step re-derives every group's next rung,
    # which is O(groups) per upgrade and irrelevant at this scale.
    while True:
        best_key: str | None = None
        best_rung: QuantType | None = None
        best_ratio = 0.0
        best_delta = 0
        for key, rungs in options.items():
            if not rungs:
                continue
            cur = chosen[key]
            idx = rungs.index(cur) if cur in rungs else -1
            if idx < 0 or idx + 1 >= len(rungs):
                continue
            nxt = rungs[idx + 1]
            delta_bytes = cost(key, nxt) - cost(key, cur)
            if delta_bytes <= 0 or spent + delta_bytes > budget_bytes:
                continue
            g = by_key[key]
            delta_err = (cur.rmse**2 - nxt.rmse**2) * g.importance
            ratio = delta_err / delta_bytes
            if ratio > best_ratio:
                best_key, best_rung, best_ratio, best_delta = key, nxt, ratio, delta_bytes
        if best_key is None:
            break
        chosen[best_key] = best_rung  # type: ignore[assignment]
        spent += best_delta

    choices = tuple(
        Choice(group=g, quant=chosen[g.key], n_bytes=cost(g.key, chosen[g.key]))
        for g in movable
    )
    return Plan(choices=choices, budget_bytes=budget_bytes, fixed_bytes=fixed)


def render_tensor_type_file(plan_: Plan) -> str:
    """Emit a `--tensor-type-file` for llama-quantize.

    Two constraints from llama.cpp, both of which bite silently if ignored.

    **No comments, no headers, nothing but assignments.** The parser is
    `while (file >> arg) parse_tensor_type(arg)` — every whitespace-separated
    token is required to be `pattern=TYPE`, so a leading `# provenance` line
    aborts the whole run with `malformed tensor type '#'`. Provenance belongs
    beside the file, not in it.

    **Patterns are matched with `std::regex_search`, which is unanchored.** A
    bare `ffn_gate` would also claim `ffn_gate_exps`, `ffn_gate_shexp` and the
    `ffn_gate_inp` router. Every pattern a Group carries is anchored for that
    reason; this only formats them.
    """
    return "".join(
        f"{c.group.pattern}={c.quant.name}\n"
        for c in sorted(plan_.choices, key=lambda c: c.group.key)
    )


def anchored(name: str) -> str:
    """An exact-match pattern for one tensor name."""
    return f"^{re.escape(name)}$"


def role_pattern(role: str) -> str:
    """A pattern matching one role across every layer, and nothing else."""
    return rf"^(blk\.\d+\.)?{re.escape(role)}$"


def normalize_energy_per_role(groups: list[Group]) -> list[Group]:
    """Rescale each group's energy against its own role's mean.

    Makes energy answer "is this layer hotter than its siblings?" (valid) instead
    of "is this tensor hotter than that unrelated one?" (not valid without σ_W —
    see the module docstring). A role whose layers are uniform comes out all 1.0
    and is then allocated on parameters and floors alone, which is the honest
    outcome when the imatrix has nothing role-relative to say.
    """
    totals: dict[str, tuple[float, int]] = {}
    for g in groups:
        role = g.role or g.key
        weighted, params = totals.get(role, (0.0, 0))
        totals[role] = (weighted + g.energy * g.n_params, params + g.n_params)

    out = []
    for g in groups:
        role = g.role or g.key
        weighted, params = totals[role]
        mean = weighted / params if params else 0.0
        out.append(replace(g, energy=g.energy / mean if mean > 0 else 1.0))
    return out


def regroup_by_role(groups: list[Group]) -> list[Group]:
    """Collapse per-tensor groups into one group per role.

    Importance becomes the parameter-weighted mean, so a role's energy reflects
    where its parameters actually are rather than treating a 40-layer stack as
    40 equal votes.
    """
    merged: dict[str, Group] = {}
    for g in groups:
        prev = merged.get(g.key)
        if prev is None:
            merged[g.key] = g
            continue
        total = prev.n_params + g.n_params
        merged[g.key] = replace(
            prev,
            n_params=total,
            n_tensors=prev.n_tensors + g.n_tensors,
            energy=(prev.energy * prev.n_params + g.energy * g.n_params) / total,
        )
    return list(merged.values())
