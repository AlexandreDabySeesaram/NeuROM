"""Mid-point (one-point) quadrature rule on the 2D reference triangle."""

import torch

from neurom.reference_cells import TRIANGLE
from neurom.quadratures.quadrature_rule import QuadratureRule


class MidPoint2D(QuadratureRule):
    """1-point midpoint quadrature rule on the 2-D reference triangle.

    The single quadrature point is placed at the centroid of the reference
    triangle (barycentric coordinates ``[1/3, 1/3, 1/3]``).  The weight
    equals the measure of the reference triangle (area ``0.5``).

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(1, 3)``.
        weights_ref (torch.Tensor): Integration weight of shape ``(1,)``.
    """

    def __init__(self):
        """Initialise the midpoint rule on the reference triangle."""
        ref = TRIANGLE

        # midpoint barycentric
        points = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])  # (1,3)

        weights = ref.measure[None]  # (1,)

        super().__init__(ref, points, weights)
