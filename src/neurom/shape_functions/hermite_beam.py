"""Hermite P3 shape functions on the reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.shape_functions.shape_function import ShapeFunction


class HermiteBeam(ShapeFunction):
    """Hermite beam shape function defined on the 1-D reference bar ``[-1, 1]``.

    The four nodal basis functions are:

    .. math::

        H_1(\\xi) = \\tfrac{1}{4}(1 - \\xi)^2(2+\\xi), \\quad
        H_2(\\xi) = \\tfrac{1}{4}(1 - \\xi)^2(\\xi + 1), \\quad
        H_3(\\xi) = \\tfrac{1}{4}(1 + \\xi)^2(2-\\xi), \\quad
        H_4(\\xi) = \\tfrac{1}{4}(1 + \\xi)^2(\\xi - 1).

    """

    def __init__(self):
        """Initialise using the standard ``Bar`` reference element."""
        super().__init__(Bar())

    def N(self, xi):
        """Evaluate the four Hermite shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, 4)``.
        """
        xi0 = xi[..., 0]

        return torch.stack(
            [
                0.25 * (2 + xi0 ) * (1 - xi0)**2,
                0.25 * (xi0 + 1.0) * (1 - xi0)**2,
                0.25 * (2 - xi0) * (1 + xi0)**2,
                0.25 * (xi0 - 1.0) * (1 + xi0)**2,
            ],
            dim=-1,
        )
