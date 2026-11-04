"""Invariant tests common to all quadrature rules.

The quadrature weights of a rule must sum to the measure of its reference
element, so that integrating the constant 1 over the reference element is
exact.
"""

import pytest
import torch

from neurom.quadratures import MidPoint1D, TwoPoints1D, MidPoint2D, ThreePoints2D

torch.set_default_dtype(torch.float32)

relative_tolerance = 1e-6


@pytest.mark.parametrize(
    "quad", [MidPoint1D(), TwoPoints1D(), MidPoint2D(), ThreePoints2D()]
)
def test_weights_sum_to_reference_measure(quad):
    """
    Test that the quadrature weights sum to the reference element measure.
    """
    assert quad.weights().sum().item() == pytest.approx(
        quad.reference_element.measure.item(), rel=relative_tolerance
    )


@pytest.mark.parametrize(
    "quad", [MidPoint1D(), TwoPoints1D(), MidPoint2D(), ThreePoints2D()]
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
