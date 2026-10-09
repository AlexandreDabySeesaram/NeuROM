"""Tests for the cubic Hermite shape functions on the reference bar."""

import pytest
import torch

from neurom.math import jacobian
from neurom.shape_functions import CubicHermiteBar

torch.set_default_dtype(torch.float32)

relative_tolerance = 1e-6


@pytest.mark.parametrize(
    "xi,expected",
    [
        (-1.0, [1.0, 0.0, 0.0, 0.0]),
        (1.0, [0.0, 0.0, 1.0, 0.0]),
        (0.0, [0.5, 0.25, 0.5, -0.25]),
        (0.5, [0.15625, 0.09375, 0.84375, -0.28125]),
    ],
)
def test_values(xi, expected):
    """
    Test the shape function values at the nodes and at interior points.
    """
    N = CubicHermiteBar().N(torch.tensor([[[xi]]]))  # (1, 1, 4)

    assert N.shape == (1, 1, 4)
    assert N.flatten().tolist() == pytest.approx(expected, rel=relative_tolerance, abs=1e-12)


@pytest.mark.parametrize(
    "xi,expected",
    [
        (-1.0, [0.0, 1.0, 0.0, 0.0]),
        (1.0, [0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_nodal_derivatives(xi, expected):
    """
    Test that H_2 and H_4 carry the slope at the end nodes: dH/dxi is one for
    the slope function of the node and zero for the others.
    """
    xi_t = torch.tensor([[[xi]]], requires_grad=True)  # (1, 1, 1)
    dN = jacobian(xi_t, CubicHermiteBar().N(xi_t))  # (1, 1, 4, 1)

    assert dN.shape == (1, 1, 4, 1)
    assert dN.flatten().tolist() == pytest.approx(expected, rel=relative_tolerance, abs=1e-12)


def test_value_functions_partition_of_unity():
    """
    Test that the value functions H_1 and H_3 sum to one anywhere in the element.
    """
    xi = torch.linspace(-1.0, 1.0, 7).reshape(1, 7, 1)
    N = CubicHermiteBar().N(xi)  # (1, 7, 4)

    assert (N[..., 0] + N[..., 2]).flatten().tolist() == pytest.approx(
        [1.0] * 7, rel=relative_tolerance
    )
