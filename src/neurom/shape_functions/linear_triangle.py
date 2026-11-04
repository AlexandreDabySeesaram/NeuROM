"""Linear shape functions on the reference triangle."""

import torch

from neurom.reference_elements.triangle import Triangle
from neurom.shape_functions.shape_function import ShapeFunction


class LinearTriangle(ShapeFunction):
    """Linear (P1) shape function defined on the 2-D reference triangle.

    The reference triangle has nodes at ``(0, 0)``, ``(1, 0)``, and ``(0, 1)``.
    The three nodal basis functions are:

    .. math::

        N_1(\\xi) = 1 - \\xi_0 - \\xi_1, \\quad
        N_2(\\xi) = \\xi_0, \\quad
        N_3(\\xi) = \\xi_1.
    """

    def __init__(self):
        """Initialise using the standard ``Triangle`` reference element."""
        super().__init__(Triangle())

    def N(self, xi):
        """Evaluate the three linear shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)`` with ``dim_ref == 2``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, 3)``.
        """
        xi0 = xi[..., 0]
        xi1 = xi[..., 1]

        return torch.stack(
            [1.0 - xi0 - xi1, xi0, xi1],
            dim=-1,
        )
