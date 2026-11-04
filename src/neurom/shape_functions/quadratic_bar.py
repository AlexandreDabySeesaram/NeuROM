"""Quadratic shape functions on the reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.shape_functions.shape_function import ShapeFunction


class QuadraticBar(ShapeFunction):
    """Quadratic (P2) shape function defined on the 1-D reference bar ``[-1, 1]``.

    The three nodal basis functions for the standard Lagrange P2 element are:

    .. math::

        N_1(\\xi) = \\tfrac{1}{2}\\xi(\\xi - 1), \\quad
        N_2(\\xi) = 1 - \\xi^2, \\quad
        N_3(\\xi) = \\tfrac{1}{2}\\xi(\\xi + 1).

    ``N_1`` and ``N_3`` are associated with the end nodes at ``\\xi = -1`` and
    ``\\xi = 1`` respectively; ``N_2`` is associated with the mid-node at
    ``\\xi = 0``.
    """

    def __init__(self):
        """Initialise using the standard ``Bar`` reference element."""
        super().__init__(Bar())

    def N(self, xi_q):
        """Evaluate the three quadratic shape functions at reference coordinates.

        Args:
            xi_q (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)`` with ``dim_ref == 1``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, 3)``.
        """

        xi = xi_q[..., 0]

        return torch.stack(
            [
                0.5 * xi * (xi - 1.0),
                1.0 - xi**2,
                0.5 * xi * (xi + 1.0),
            ],
            dim=-1,
        )
