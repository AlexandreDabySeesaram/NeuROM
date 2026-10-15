"""Three-point quadrature rule on the 2D reference triangle."""

import torch

from neurom.reference_cells import TRIANGLE
from neurom.quadratures.quadrature_rule import QuadratureRule


class ThreePoints2D(QuadratureRule):
    """3-point quadrature rule on the 2-D reference triangle.

    The three quadrature points are placed at the midpoints of the triangle
    edges.  In barycentric coordinates the points are:

    - ``[1/6, 1/6, 2/3]``
    - ``[1/6, 2/3, 1/6]``
    - ``[2/3, 1/6, 1/6]``

    Each weight equals one third of the reference triangle area.  This rule
    integrates polynomials up to degree 2 exactly.

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(3, 3)``.
        weights_ref (torch.Tensor): Integration weights of shape ``(3,)``.
    """

    def __init__(self):
        """Initialise the 3-point rule on the reference triangle."""
        ref = TRIANGLE

        # three points barycentrics barycentric
        points = torch.tensor(
            [
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            ]
        )  # (3,3)

        weights = ref.measure / 3.0 * torch.ones(3)  # (3,)

        super().__init__(ref, points, weights)
