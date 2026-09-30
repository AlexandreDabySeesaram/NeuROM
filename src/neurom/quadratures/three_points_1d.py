"""Three-point Gauss quadrature rule on the 1D reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.quadratures.quadrature_rule import QuadratureRule


class ThreePoints1D(QuadratureRule):
    """3-point Gauss–Legendre quadrature rule on the 1-D reference bar ``[-1, 1]``.

    The three Gauss points are located at :math:`\\xi = 0, \\pm\\sqrt{3/5}`.
    In barycentric coordinates the points are stored as rows of shape
    ``(3, 2)``, with weights :math:`5/9, 8/9, 5/9` times half the reference
    element measure.  This rule integrates polynomials up to degree 5 exactly.

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(3, 2)``.
        weights_ref (torch.Tensor): Integration weights of shape ``(3,)``.
    """

    def __init__(self):
        """Initialise the 3-point Gauss rule on the standard ``Bar`` element."""
        ref = Bar()
        super().__init__(ref)

        # Gauss points in barycentric coordinates
        s = torch.sqrt(torch.tensor(3.0 / 5.0))
        a = 0.5 * (1.0 - s)
        b = 0.5 * (1.0 + s)

        points = torch.stack(
            [
                torch.tensor([b, a]),
                torch.tensor([0.5, 0.5]),
                torch.tensor([a, b]),
            ]
        )  # (3,2)

        weights = (
            0.5 * ref.measure * torch.tensor([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        )  # (3,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
