"""WP9 stage A: the vector-valued-basis plumbing (ShapeFunction.n_ref + the contraction).

A vector basis returns ``N(xi) = (*batch, n_ref, gdim)`` -- each basis function is itself a
``gdim``-vector (RT0). This exercises, with a tiny toy basis (so it's independent of RT0's
correctness): (1) ``n_ref`` must be the basis-function count (``shape[-2]``), not ``gdim``
(``shape[-1]``); (2) the interpolation einsums contract the basis axis and keep the vector axis.
"""

import pytest
import torch

from neurom.reference_cell import TRIANGLE
from neurom.shape_functions.shape_function import ShapeFunction
from neurom.shape_functions import LinearTriangle

F64 = torch.float64


class _ToyVectorBasis(ShapeFunction):
    """3 basis functions on the reference triangle, each a 2-vector -> N = (*batch, 3, 2).

    Deliberately ``n_ref=3 != gdim=2`` so the two axes can't be confused.
    """

    def __init__(self):
        super().__init__(TRIANGLE)

    def N(self, xi):
        l0 = 1.0 - xi[..., 0] - xi[..., 1]
        l1 = xi[..., 0]
        l2 = xi[..., 1]
        z = torch.zeros_like(l0)
        H0 = torch.stack([l0, z], dim=-1)
        H1 = torch.stack([z, l1], dim=-1)
        H2 = torch.stack([l2, l2], dim=-1)
        return torch.stack([H0, H1, H2], dim=-2)  # (*batch, n_ref=3, gdim=2)

    @property
    def n_ref(self):
        return 3


def test_scalar_basis_n_ref_is_the_trailing_axis():
    """The default (probe) n_ref is correct for a scalar basis (P1 triangle -> 3)."""
    assert LinearTriangle().n_ref == 3


def test_vector_basis_n_ref_is_the_basis_count_not_gdim():
    """A vector basis reports n_ref = basis-function count (3), not gdim (= N.shape[-1] = 2)."""
    toy = _ToyVectorBasis()
    probe = TRIANGLE.simplex[:1].to(F64)
    assert toy.N(probe).shape[-1] == 2  # trailing axis is gdim -- what the default probe would read
    assert toy.n_ref == 3  # ... but the overridden n_ref is the basis count


def test_vector_basis_contraction_keeps_the_vector_axis():
    """Both interpolation einsums contract the basis axis n and keep the trailing gdim via '...'."""
    toy = _ToyVectorBasis()
    # An interior point (the centroid) so every field component is non-zero -> relative tolerance.
    centroid = torch.tensor([[1.0 / 3, 1.0 / 3]], dtype=F64)  # (1, 2)
    xi = centroid.expand(2, 3, -1)  # (N_e=2, N_q=3, 2)
    N = toy.N(xi)  # (2, 3, 3, 2) = (N_e, N_q, n_ref, gdim)
    assert N.shape == (2, 3, 3, 2)
    torch.manual_seed(0)
    u = torch.randn(2, 3, dtype=F64)  # (N_e, n_ref) scalar coefficients, value_shape ()

    # FieldInterpolator.at_reference pattern: contract basis axis n, keep gdim via '...'.
    out_q = torch.einsum("en...,eqn...->eq...", u, N)
    assert out_q.shape == (2, 3, 2)  # (N_e, N_q, gdim) -- vector field value

    # PointWiseInterpolator.at_position pattern (one point per element).
    N_pt = toy.N(centroid.expand(2, -1))  # (2, 3, 2) one point per elem
    out_pt = torch.einsum("en...,en...->e...", u, N_pt)
    assert out_pt.shape == (2, 2)  # (N_pts, gdim)

    # Both contractions equal the explicit vector interpolant sum_a u_a H_a (no einsum).
    expected = sum(u[:, a : a + 1] * N_pt[:, a, :] for a in range(3))  # (2, 2)
    assert out_pt.numpy() == pytest.approx(expected.numpy(), rel=1e-12)
    assert out_q[:, 0, :].numpy() == pytest.approx(expected.numpy(), rel=1e-12)
