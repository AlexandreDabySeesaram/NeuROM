"""Mid-point (one-point) quadrature rule on the 1D reference bar."""

import torch

from neurom.reference_cells import BAR
from neurom.quadratures.quadrature_rule import QuadratureRule


class MidPoint1D(QuadratureRule):
    """1-point midpoint quadrature rule on the 1-D reference bar ``[-1, 1]``.

    The single quadrature point is placed at the midpoint of the bar
    (barycentric coordinates ``[0.5, 0.5]``), i.e. ``\\xi = 0``.  The weight
    equals the measure of the reference element (length ``2``).

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(1, 2)``.
        weights_ref (torch.Tensor): Integration weight of shape ``(1,)``.
    """

    def __init__(self):
        """Initialise the midpoint rule on the reference bar."""
        ref = BAR

        # midpoint barycentric
        points = torch.tensor([[0.5, 0.5]])  # (1,2)

        weights = ref.measure[None]  # (1,)

        super().__init__(ref, points, weights)
