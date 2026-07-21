"""Two-point Gauss quadrature rule on the 1D reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.quadratures.quadrature_rule import QuadratureRule


class TwoPoints1D(QuadratureRule):
    """2-point Gauss–Legendre quadrature rule on the 1-D reference bar ``[-1, 1]``.

    The two Gauss points are located at :math:`\\xi = \\pm 1/\\sqrt{3}`.
    In barycentric coordinates the points are stored as rows of shape
    ``(2, 2)``, with each weight equal to half the reference element measure
    (i.e. ``1.0``).  This rule integrates polynomials up to degree 3 exactly.

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(2, 2)``.
        weights_ref (torch.Tensor): Integration weights of shape ``(2,)``.
    """

    def __init__(self):
        """Initialise the 2-point Gauss rule on the standard ``Bar`` element."""
        ref = Bar()
        super().__init__(ref)

        # Gauss points in barycentric coordinates
        a = 0.5 * (1.0 - 1.0 / torch.sqrt(torch.tensor(3.0)))
        b = 0.5 * (1.0 + 1.0 / torch.sqrt(torch.tensor(3.0)))

        points = torch.stack(
            [
                torch.tensor([b, a]),
                torch.tensor([a, b]),
            ]
        )  # (2,2)

        weights = 0.5 * ref.measure * torch.ones(2)  # (2,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
