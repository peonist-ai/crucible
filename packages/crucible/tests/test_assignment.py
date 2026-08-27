"""Optimal assignment — the scipy replacement REAM's neuron alignment depends on.

Correctness here is checked by the property that defines the problem: the total
cost must be minimal, and every row must get a distinct column. We assert
against brute force on small matrices rather than against a reference
implementation, so these tests hold with no scientific stack installed.
"""

import itertools

import pytest
import torch

from crucible.methods.assignment import linear_sum_assignment


def brute_force_min_cost(cost: torch.Tensor) -> float:
    """Exhaustive optimum. Only tractable for tiny matrices.

    When there are more rows than columns only `m` of the rows get assigned,
    so transpose first and the shorter side becomes the one to permute over.
    Optimal cost is invariant under transpose.
    """
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
    n, m = cost.shape
    best = float("inf")
    for cols in itertools.permutations(range(m), n):
        total = sum(cost[i, c].item() for i, c in enumerate(cols))
        best = min(best, total)
    return best


def total(cost, rows, cols):
    return cost[rows, cols].sum().item()


class TestOptimality:
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4), (3, 5), (5, 3)])
    def test_matches_brute_force(self, shape):
        torch.manual_seed(shape[0] * 100 + shape[1])
        cost = torch.rand(*shape, dtype=torch.float64) * 10

        rows, cols = linear_sum_assignment(cost)

        assert total(cost, rows, cols) == pytest.approx(brute_force_min_cost(cost))

    def test_handles_ties(self):
        # Integer costs produce many equal-cost assignments; the solver must
        # still land on an optimal one rather than tripping over the ties.
        torch.manual_seed(7)
        cost = torch.randint(0, 3, (6, 6)).double()

        rows, cols = linear_sum_assignment(cost)

        assert total(cost, rows, cols) == pytest.approx(brute_force_min_cost(cost))

    def test_handles_negative_costs(self):
        torch.manual_seed(11)
        cost = torch.rand(5, 5, dtype=torch.float64) * 10 - 5

        rows, cols = linear_sum_assignment(cost)

        assert total(cost, rows, cols) == pytest.approx(brute_force_min_cost(cost))

    def test_known_optimum(self):
        # Identity is the only sane matching here.
        cost = torch.tensor(
            [[1.0, 9.0, 9.0], [9.0, 1.0, 9.0], [9.0, 9.0, 1.0]], dtype=torch.float64
        )

        rows, cols = linear_sum_assignment(cost)

        assert cols.tolist() == [0, 1, 2]
        assert total(cost, rows, cols) == pytest.approx(3.0)


class TestContract:
    def test_assignment_is_a_permutation(self):
        torch.manual_seed(3)
        cost = torch.rand(20, 20, dtype=torch.float64)

        rows, cols = linear_sum_assignment(cost)

        assert rows.tolist() == list(range(20))
        assert sorted(cols.tolist()) == list(range(20))

    def test_wide_matrix_uses_distinct_columns(self):
        torch.manual_seed(5)
        cost = torch.rand(4, 9, dtype=torch.float64)

        rows, cols = linear_sum_assignment(cost)

        assert rows.tolist() == list(range(4))
        assert len(set(cols.tolist())) == 4

    def test_tall_matrix_returns_sorted_rows(self):
        # n > m is solved transposed internally; row_ind must still come back
        # ascending, matching scipy's documented contract.
        torch.manual_seed(6)
        cost = torch.rand(9, 4, dtype=torch.float64)

        rows, cols = linear_sum_assignment(cost)

        assert rows.tolist() == sorted(rows.tolist())
        assert len(rows) == 4
        assert len(set(cols.tolist())) == 4

    def test_returns_long_tensors_on_input_device(self):
        cost = torch.rand(3, 3)

        rows, cols = linear_sum_assignment(cost)

        assert rows.dtype == torch.long
        assert cols.dtype == torch.long
        assert rows.device == cost.device

    def test_float32_input_accepted(self):
        # Callers pass float32; the solver promotes internally because the dual
        # updates accumulate and drift at single precision.
        cost = torch.rand(6, 6, dtype=torch.float32)

        rows, cols = linear_sum_assignment(cost)

        assert sorted(cols.tolist()) == list(range(6))


class TestRejectsBadInput:
    def test_empty(self):
        rows, cols = linear_sum_assignment(torch.empty(0, 0))
        assert len(rows) == 0
        assert len(cols) == 0

    def test_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_sum_assignment(torch.rand(3, 3, 3))

    def test_nan_rejected(self):
        cost = torch.rand(3, 3, dtype=torch.float64)
        cost[1, 1] = float("nan")

        with pytest.raises(ValueError, match="infinite or NaN"):
            linear_sum_assignment(cost)

    def test_inf_rejected(self):
        cost = torch.rand(3, 3, dtype=torch.float64)
        cost[0, 2] = float("inf")

        with pytest.raises(ValueError, match="infinite or NaN"):
            linear_sum_assignment(cost)
