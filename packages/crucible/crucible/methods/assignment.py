"""Optimal rectangular assignment (Hungarian / Jonker-Volgenant).

Replaces `scipy.optimize.linear_sum_assignment`, the only thing crucible used
scipy for. Carrying a large scientific stack for one function is a poor trade
when the function is ~60 lines.

Shortest-augmenting-path formulation with dual potentials. The inner loop over
columns is vectorized with numpy — which torch already depends on, so this adds
nothing to the dependency tree. Measured at REAM's real matrix size (704x704):
~170 ms versus scipy's ~11 ms. That 16x is worth it here because it runs inside
a method that already takes ~20 minutes, and it buys zero required dependencies.

Note on determinism: when several assignments tie for optimal, this returns a
specific one, and it is not necessarily the one scipy would pick. Both are
optimal — equal total cost — but the permutations can differ. That is precisely
why we do *not* opportunistically use scipy when it happens to be installed:
identical inputs must produce identical compressed models regardless of which
packages a machine has lying around.
"""

from __future__ import annotations

import numpy as np
import torch


def linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the linear sum assignment problem.

    Assigns rows to distinct columns minimizing total cost.

    Args:
        cost: 2-D tensor [n, m]. Transposed internally when n > m.

    Returns:
        (row_ind, col_ind) with cost[row_ind, col_ind].sum() minimal, row_ind
        ascending — matching scipy's contract so call sites port unchanged.
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {tuple(cost.shape)}")

    device = cost.device
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()

    a = cost.detach().to(torch.float64).cpu().numpy()
    if not np.isfinite(a).all():
        raise ValueError("cost matrix contains infinite or NaN entries")

    transposed = a.shape[0] > a.shape[1]
    if transposed:
        a = a.T

    col_for_row = _solve(a)
    rows = torch.arange(a.shape[0], dtype=torch.long, device=device)
    cols = torch.as_tensor(col_for_row, dtype=torch.long, device=device)

    if transposed:
        # Our "rows" were the caller's columns — swap back and re-sort.
        order = torch.argsort(cols)
        return cols[order], rows[order]
    return rows, cols


def _solve(a: np.ndarray) -> np.ndarray:
    """Core solver for n <= m. Returns the column assigned to each row."""
    n, m = a.shape
    inf = np.inf

    # 1-indexed, with slot 0 as the "free" sentinel — that is what keeps the
    # augmenting-path bookkeeping branch-free.
    u = np.zeros(n + 1)                       # row potentials
    v = np.zeros(m + 1)                       # column potentials
    p = np.zeros(m + 1, dtype=np.int64)       # p[j] = row matched to column j
    way = np.zeros(m + 1, dtype=np.int64)     # predecessor column on the path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]

            # Reduced cost of row i0 against every column not yet on the path.
            free = ~used[1:]
            reduced = a[i0 - 1] - u[i0] - v[1:]
            improved = free & (reduced < minv[1:])
            minv[1:][improved] = reduced[improved]
            way[1:][improved] = j0

            candidates = np.where(free, minv[1:], inf)
            arg = int(np.argmin(candidates))
            delta = candidates[arg]
            if not np.isfinite(delta):
                raise ValueError("cost matrix admits no feasible assignment")
            j1 = arg + 1

            # Shift the duals so edges already on the path stay tight.
            np.add.at(u, p[used], delta)
            v[used] -= delta
            minv[~used] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Walk the augmenting path back, flipping matches along the way.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    col_for_row = np.zeros(n, dtype=np.int64)
    matched = p[1:] > 0
    col_for_row[p[1:][matched] - 1] = np.nonzero(matched)[0]
    return col_for_row
