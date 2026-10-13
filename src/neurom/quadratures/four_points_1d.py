"""Four-point Gauss quadrature rule on the 1D reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.quadratures.quadrature_rule import QuadratureRule


class FourPoints1D(QuadratureRule):
    """4-point Gauss–Legendre quadrature rule on the 1-D reference bar ``[-1, 1]``.

    The four Gauss points are located at
    :math:`\\xi = \\pm\\sqrt{3/7 \\mp (2/7)\\sqrt{6/5}}`, with weights
    :math:`(18 \\pm \\sqrt{30})/36` (the larger weight on the inner points)
    times half the reference element measure.  In barycentric coordinates the
    points are stored as rows of shape ``(4, 2)``.  This rule integrates
    polynomials up to degree 7 exactly.

    Attributes:
        points_barycentric (torch.Tensor): Barycentric coordinates of shape
            ``(4, 2)``.
        weights_ref (torch.Tensor): Integration weights of shape ``(4,)``.
    """

    def __init__(self):
        """Initialise the 4-point Gauss rule on the standard ``Bar`` element."""
        ref = Bar()
        super().__init__(ref)

        # Gauss points on [-1, 1]: inner (i) and outer (o) pairs
        r = 2.0 / 7.0 * torch.sqrt(torch.tensor(6.0 / 5.0))
        xi_i = torch.sqrt(3.0 / 7.0 - r)
        xi_o = torch.sqrt(3.0 / 7.0 + r)
        xi = torch.stack([-xi_o, -xi_i, xi_i, xi_o])

        # Barycentric coordinates, same convention as ThreePoints1D
        points = torch.stack([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], dim=-1)  # (4,2)

        sqrt30 = torch.sqrt(torch.tensor(30.0))
        w_i = (18.0 + sqrt30) / 36.0
        w_o = (18.0 - sqrt30) / 36.0
        weights = 0.5 * ref.measure * torch.stack([w_o, w_i, w_i, w_o])  # (4,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
