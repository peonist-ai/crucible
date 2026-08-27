"""Per-role sensitivity, calibrated from measured KL-divergence.

The term an importance matrix cannot supply. An imatrix measures how large a
tensor's *inputs* are, which on a normalized transformer clusters by position
relative to the nearest RMSNorm — everything reading a normed hidden state comes
back at ~0.9 whatever it does, and `ssm_out`, sitting behind the delta-net, comes
back at 3e-4 whether or not it matters. Weight scale does not rescue the
comparison either: measured, σ_W spans 2.4x across roles while energy spans
3,317x, so including it widens the gap.

What actually answers "how much does degrading this role move the output
distribution" is degrading it and measuring. `scripts/measure_role_sensitivity.py`
does that: hold every role at a baseline encoding, drop one role to a probe
encoding, measure KL-divergence against the f16 model, repeat. The difference
from the all-baseline run is that role's contribution.

Converting a measured ΔKLD into a coefficient the planner can use: the objective
models a role's error as

    error = s_role × rmse(type)² × n_params

so a probe that moves one role from `baseline` to `probe` should produce

    ΔKLD ≈ s_role × (rmse(probe)² − rmse(baseline)²) × n_params

and therefore

    s_role = ΔKLD ÷ ((rmse(probe)² − rmse(baseline)²) × n_params)

Coefficients are normalised to a parameter-weighted mean of 1, so a table only
ever redistributes budget — swapping one in never silently inflates or shrinks
the total the solver thinks it is spending.

The table is reusable: it is a property of an architecture, not of a budget or a
prune ratio, so one measurement serves every tier and every REAP ratio for a
model family.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoleSensitivity:
    """What one probe measured."""

    role: str
    n_params: int
    delta_kld: float
    coefficient: float


@dataclass(frozen=True)
class SensitivityTable:
    """Measured per-role sensitivity, normalised to a mean of 1."""

    baseline_type: str
    probe_type: str
    baseline_kld: float
    roles: dict[str, RoleSensitivity]
    source: str = ""

    def coefficient(self, role: str | None) -> float:
        """The multiplier for `role`, or 1.0 if it was never probed."""
        if role is None:
            return 1.0
        entry = self.roles.get(role)
        return entry.coefficient if entry else 1.0

    @property
    def unmeasured_note(self) -> str:
        return (
            f"{len(self.roles)} roles measured "
            f"({self.baseline_type} -> {self.probe_type}, "
            f"baseline KLD {self.baseline_kld:.6f})"
        )


def load(path: str | Path) -> SensitivityTable:
    """Read a table written by `scripts/measure_role_sensitivity.py`.

    Raises on a malformed or degenerate table rather than falling back to
    uniform: a sensitivity file that silently means nothing is worse than none,
    because the planner would report itself as measurement-driven while running
    on the same priors it always had.
    """
    from crucible.quant_types import QUANT_TYPES

    path = Path(path)
    doc = json.loads(path.read_text())

    for key in ("baseline_type", "probe_type", "baseline_kld", "roles"):
        if key not in doc:
            raise ValueError(f"{path}: sensitivity table is missing {key!r}")

    baseline, probe = doc["baseline_type"], doc["probe_type"]
    for name in (baseline, probe):
        if name not in QUANT_TYPES:
            raise ValueError(f"{path}: unknown quantization type {name!r}")

    # The probe must be strictly worse than the baseline, or the denominator is
    # zero or negative and every coefficient comes out meaningless.
    spread = QUANT_TYPES[probe].rmse ** 2 - QUANT_TYPES[baseline].rmse ** 2
    if spread <= 0:
        raise ValueError(
            f"{path}: probe {probe!r} is not more distorting than baseline "
            f"{baseline!r} — the probe must degrade the role being measured"
        )

    raw: dict[str, tuple[int, float, float]] = {}
    for role, entry in doc["roles"].items():
        n_params = int(entry["n_params"])
        delta = float(entry["delta_kld"])
        if n_params <= 0:
            raise ValueError(f"{path}: {role} has n_params={n_params}")
        # A probe that came back at or below the baseline measured nothing
        # usable. Clamp to a small positive rather than zero, so the role is
        # ranked last instead of being handed an infinitely cheap upgrade.
        raw[role] = (n_params, delta, max(delta, 1e-9) / (spread * n_params))

    if not raw:
        raise ValueError(f"{path}: no roles in the table")

    total_params = sum(v[0] for v in raw.values())
    weighted_mean = sum(v[2] * v[0] for v in raw.values()) / total_params
    if weighted_mean <= 0:
        raise ValueError(f"{path}: all coefficients are zero — nothing was measured")

    roles = {
        role: RoleSensitivity(
            role=role, n_params=n, delta_kld=d, coefficient=c / weighted_mean
        )
        for role, (n, d, c) in raw.items()
    }
    return SensitivityTable(
        baseline_type=baseline,
        probe_type=probe,
        baseline_kld=float(doc["baseline_kld"]),
        roles=roles,
        source=str(path),
    )
