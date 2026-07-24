"""Unit tests for the degeneracy diagnostics, against a stub decomposition.

The helpers only ever call ``monoms[m][k].full_values()``, so the stub is a
nested list of tensors. Deliberately no FEM: whether the correlation formula is
right is independent of what the nodal values mean.
"""

import math

import pytest
import torch

from neurom.training import diagnostics


class StubField:
    def __init__(self, values):
        self.values = torch.as_tensor(values, dtype=torch.float64)

    def full_values(self):
        return self.values


class StubDecomposition:
    """``modes`` is a list of modes, each a list of per-axis nodal vectors."""

    def __init__(self, modes):
        self.monoms = [[StubField(axis) for axis in mode] for mode in modes]
        self.n_modes_truncated = len(modes)


def test_amplitude_is_the_product_of_the_axis_norms():
    decomposition = StubDecomposition([[[3.0, 4.0], [0.0, 2.0]]])
    values = diagnostics.monom_values(decomposition, 0)

    assert diagnostics.amplitude(values) == pytest.approx(5.0 * 2.0)


def test_parallel_modes_correlate_to_one_whatever_their_scales():
    # CP factors are defined only up to a per-axis scale, so a mode and a
    # rescaled copy of it are THE SAME mode -- the correlation must say so.
    decomposition = StubDecomposition(
        [[[1.0, 2.0], [3.0, 1.0]], [[-7.0, -14.0], [0.3, 0.1]]]
    )

    assert diagnostics.max_correlation(decomposition, 1) == pytest.approx(1.0)


def test_a_mode_orthogonal_on_one_axis_is_uncorrelated():
    decomposition = StubDecomposition(
        [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]]]
    )

    assert diagnostics.max_correlation(decomposition, 1) == pytest.approx(0.0)


def test_the_first_mode_has_nothing_to_duplicate():
    decomposition = StubDecomposition([[[1.0, 2.0], [3.0, 1.0]]])

    assert diagnostics.max_correlation(decomposition, 0) == 0.0


def test_a_vanishing_axis_makes_the_mode_duplicate_nothing():
    # A mode that is zero on one axis is the zero tensor: the quotient is
    # undefined, and reporting 0.0 is the honest answer, not a NaN.
    decomposition = StubDecomposition(
        [[[1.0, 2.0], [3.0, 1.0]], [[0.0, 0.0], [3.0, 1.0]]]
    )

    assert diagnostics.max_correlation(decomposition, 1) == 0.0


def test_a_non_finite_mode_reports_nan_rather_than_orthogonality():
    # max(0.0, nan) is 0.0 in Python, which would claim perfect orthogonality
    # exactly when the state is garbage. The reduction must not do that.
    decomposition = StubDecomposition(
        [[[1.0, 2.0], [3.0, 1.0]], [[math.nan, 1.0], [3.0, 1.0]]]
    )

    assert math.isnan(diagnostics.max_correlation(decomposition, 1))


def test_pairwise_sees_a_duplicate_pair_that_excludes_the_newest_mode():
    # THE reason the pairwise variant exists: modes 0 and 1 are copies, while
    # the newest mode is orthogonal to both. A newest-mode-only measure reports
    # a clean 0.0 on a decomposition that has already degenerated.
    decomposition = StubDecomposition(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [5.0, 5.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )

    assert diagnostics.max_correlation(decomposition, 2) == pytest.approx(0.0)
    assert diagnostics.max_pairwise_correlation(decomposition) == pytest.approx(1.0)


def test_pairwise_has_no_pair_to_look_at_with_a_single_mode():
    decomposition = StubDecomposition([[[1.0, 2.0], [3.0, 1.0]]])

    assert diagnostics.max_pairwise_correlation(decomposition) == 0.0
