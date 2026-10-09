"""Cubic Hermite shape functions on the reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.shape_functions.shape_function import ShapeFunction


class CubicHermiteBar(ShapeFunction):
    """Cubic Hermite shape function defined on the 1-D reference bar ``[-1, 1]``.

    The four nodal basis functions are:

    .. math::

        H_1(\\xi) = \\tfrac{1}{4}(1 - \\xi)^2(2 + \\xi)

        H_2(\\xi) = \\tfrac{1}{4}(1 - \\xi)^2(\\xi + 1)

        H_3(\\xi) = \\tfrac{1}{4}(1 + \\xi)^2(2 - \\xi)

        H_4(\\xi) = \\tfrac{1}{4}(1 + \\xi)^2(\\xi - 1)

    :math:`H_1` and :math:`H_3` carry the value at the end nodes
    :math:`\\xi = -1` and :math:`\\xi = 1` respectively; :math:`H_2` and
    :math:`H_4` carry the slope :math:`\\mathrm{d}/\\mathrm{d}\\xi` at the
    same nodes.

    The element DOFs are :math:`(w, \\mathrm{d}w/\\mathrm{d}\\xi)` at
    :math:`\\xi = -1` then at :math:`\\xi = 1`.
    """

    dof_kinds = ("value", "d1", "value", "d1")
    dof_nodes_xi = (-1.0, -1.0, 1.0, 1.0)

    def __init__(self):
        """Initialise using the standard ``Bar`` reference element."""
        super().__init__(Bar())

    def N(self, xi):
        """Evaluate the four cubic Hermite shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, 4)``.
        """
        xi0 = xi[..., 0]

        one_minus_xi_squared = (1.0 - xi0) ** 2
        one_plus_xi_squared = (1.0 + xi0) ** 2

        return torch.stack(
            [
                (2.0 + xi0) * one_minus_xi_squared,
                (xi0 + 1.0) * one_minus_xi_squared,
                (2.0 - xi0) * one_plus_xi_squared,
                (xi0 - 1.0) * one_plus_xi_squared,
            ],
            dim=-1,
        ).mul_(0.25)
