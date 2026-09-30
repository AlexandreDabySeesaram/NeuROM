"""Invariant tests common to all quadrature rules.

The quadrature weights of a rule must sum to the measure of its reference
element, so that integrating the constant 1 over the reference element is
exact.
"""

import pytest
import torch

from neurom.quadratures import (
    MidPoint1D,
    TwoPoints1D,
    ThreePoints1D,
    MidPoint2D,
    ThreePoints2D,
)

torch.set_default_dtype(torch.float32)

relative_tolerance = 1e-6


@pytest.mark.parametrize(
    "quad",
    [MidPoint1D(), TwoPoints1D(), ThreePoints1D(), MidPoint2D(), ThreePoints2D()],
)
def test_weights_sum_to_reference_measure(quad):
    """
    Test that the quadrature weights sum to the reference element measure.
    """
    assert quad.weights().sum().item() == pytest.approx(
        quad.reference_element.measure.item(), rel=relative_tolerance
    )


@pytest.mark.parametrize(
    "quad",
    [MidPoint1D(), TwoPoints1D(), ThreePoints1D(), MidPoint2D(), ThreePoints2D()],
)
def test_points_are_barycentric(quad):
    """
    Test that quadrature points are valid barycentric coordinates:
    components in [0, 1] summing to 1 per point.
    """
    points = quad.points()
    assert (points >= 0.0).all()
    assert (points <= 1.0).all()
    assert points.sum(dim=-1).numpy() == pytest.approx(
        torch.ones(points.shape[0]).numpy(), rel=relative_tolerance
    )


@pytest.mark.parametrize("degree", range(6))
def test_three_points_1d_integrates_polynomials_up_to_degree_5(degree):
    """
    Test that ThreePoints1D integrates x**degree exactly over [-1, 1]
    for degree <= 5.
    """
    quad = ThreePoints1D()
    x = 2.0 * quad.points()[:, 0] - 1.0
    integral = (quad.weights() * x**degree).sum()
    if degree % 2 == 0:
        assert integral.item() == pytest.approx(
            2.0 / (degree + 1), rel=relative_tolerance
        )
    else:
        assert torch.allclose(integral, torch.tensor(0.0), atol=1e-6)
