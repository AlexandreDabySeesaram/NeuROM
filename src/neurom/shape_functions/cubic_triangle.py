r"""Cubic shape functions on the reference triangle (10-node :math:`P_3`).

Ten nodal Lagrange cubics: three vertices, two interior points on each edge (at :math:`1/3` and
:math:`2/3`) and one at the centroid. In barycentric coordinates :math:`\lambda_0 = 1-\xi_0-\xi_1`,
:math:`\lambda_1 = \xi_0`, :math:`\lambda_2 = \xi_1`:

.. math::

    N_{v_i} = \tfrac12\,\lambda_i(3\lambda_i-1)(3\lambda_i-2), \qquad
    N_{e_{ij},\,i} = \tfrac92\,\lambda_i\lambda_j(3\lambda_i-1), \qquad
    N_{c} = 27\,\lambda_0\lambda_1\lambda_2 .

Columns are ordered to match the P3 DOF layout ``[v0, v1, v2, e01a, e01b, e12a, e12b, e20a, e20b,
c]`` -- vertices, then each edge's two DOFs (the one nearer the edge's first local vertex first,
following the cyclic edge convention ``(0,1),(1,2),(2,0)``), then the centroid. The basis is nodal
at those ten points, so the element is nodal (``V = I``) and the reference node coordinates passed
to the element (:data:`neurom.elements.P3_TRIANGLE`) list the same ten points in the same order.
"""

import torch

from neurom.reference_cell import TRIANGLE
from neurom.shape_functions.shape_function import ShapeFunction


class CubicTriangle(ShapeFunction):
    r"""Cubic (:math:`P_3`) nodal Lagrange basis on the 2-D reference triangle: 10 nodes.

    Three vertex nodes, six edge nodes (two per edge, at :math:`1/3` and :math:`2/3` along the
    edge) and one centroid node, in barycentric coordinates :math:`\lambda_0 = 1-\xi_0-\xi_1`,
    :math:`\lambda_1 = \xi_0`, :math:`\lambda_2 = \xi_1`. Nodal at those points, so the
    physical-to-reference transformation is the identity.
    """

    def __init__(self):
        """Initialise using the reference triangle."""
        super().__init__(TRIANGLE)

    def N(self, xi):
        r"""Evaluate the ten cubic shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(*batch, 2)``.

        Returns:
            torch.Tensor: Shape-function values, shape ``(*batch, 10)``, ordered ``[v0, v1, v2,
            e01a, e01b, e12a, e12b, e20a, e20b, c]`` -- three vertex cubics, each edge's two cubics
            (nearer the edge's first local vertex first), then the centroid bubble.
        """
        l0 = 1.0 - xi[..., 0] - xi[..., 1]
        l1 = xi[..., 0]
        l2 = xi[..., 1]
        return torch.stack(
            [
                0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0),  # v0
                0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0),  # v1
                0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0),  # v2
                4.5 * l0 * l1 * (3.0 * l0 - 1.0),  # e01, nearer v0
                4.5 * l0 * l1 * (3.0 * l1 - 1.0),  # e01, nearer v1
                4.5 * l1 * l2 * (3.0 * l1 - 1.0),  # e12, nearer v1
                4.5 * l1 * l2 * (3.0 * l2 - 1.0),  # e12, nearer v2
                4.5 * l2 * l0 * (3.0 * l2 - 1.0),  # e20, nearer v2
                4.5 * l2 * l0 * (3.0 * l0 - 1.0),  # e20, nearer v0
                27.0 * l0 * l1 * l2,  # centroid
            ],
            dim=-1,
        )
