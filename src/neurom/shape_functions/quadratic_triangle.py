"""Quadratic shape functions on the reference triangle (6-node P2)."""

import torch

from neurom.reference_elements.triangle import Triangle
from neurom.shape_functions.shape_function import ShapeFunction


class QuadraticTriangle(ShapeFunction):
    """Quadratic (P2) shape function on the 2-D reference triangle: 6 nodes.

    Three vertex nodes at :math:`(0,0)`, :math:`(1,0)`, :math:`(0,1)` and three edge-midpoint
    nodes, ordered to match the triangle's edge convention ``(0,1),(1,2),(2,0)`` -- so the
    basis columns are ``[v0, v1, v2, e01, e12, e20]``. In barycentric coordinates
    :math:`\\lambda_0 = 1 - \\xi_0 - \\xi_1`, :math:`\\lambda_1 = \\xi_0`,
    :math:`\\lambda_2 = \\xi_1`:

    .. math::

        N_{v_i} = \\lambda_i (2\\lambda_i - 1), \\qquad
        N_{e_{ij}} = 4\\,\\lambda_i \\lambda_j .

    The basis is nodal (``N_a`` = 1 at node ``a``, 0 at the others), so the physical-to-reference
    transformation is the identity.
    """

    def __init__(self):
        """Initialise using the standard ``Triangle`` reference element."""
        super().__init__(Triangle())

    def N(self, xi):
        """Evaluate the six quadratic shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates of shape ``(N_e, N_q, 2)``.

        Returns:
            torch.Tensor: Shape-function values of shape ``(N_e, N_q, 6)``, ordered
            ``[v0, v1, v2, e01, e12, e20]``.
        """
        l0 = 1.0 - xi[..., 0] - xi[..., 1]
        l1 = xi[..., 0]
        l2 = xi[..., 1]
        return torch.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ],
            dim=-1,
        )
