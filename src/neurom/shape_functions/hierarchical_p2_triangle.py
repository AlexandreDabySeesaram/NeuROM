r"""Hierarchical (non-nodal) quadratic basis on the reference triangle.

Spans the same space :math:`P_2` as :class:`~neurom.shape_functions.quadratic_triangle.QuadraticTriangle`,
but with a *hierarchical* basis -- the three linear vertex hats plus the three quadratic edge bubbles:

.. math::

    \hat\Phi = [\,\lambda_0,\ \lambda_1,\ \lambda_2,\ 4\lambda_0\lambda_1,\ 4\lambda_1\lambda_2,\
    4\lambda_2\lambda_0\,],

columns ordered ``[v0, v1, v2, e01, e12, e20]`` to match the triangle edge convention
``(0,1),(1,2),(2,0)``. This basis is **not** nodal at the P2 DOFs (a vertex hat is non-zero at the
adjacent edge midpoints), so :math:`V = \hat\ell_a(\hat\Phi_b) \ne I`; an element using it must be
declared ``nodal=False``, and its construction folds :math:`C = V^{-1}` into the tabulation (see
:class:`~neurom.shape_functions.folded_basis.FoldedBasis`), recovering the nodal P2 basis exactly.

No library element singleton ships for this basis: it exists to exercise the non-nodal folding path
(every element in the public library is nodal).
"""

import torch

from neurom.reference_cells import TRIANGLE
from neurom.shape_functions.shape_function import ShapeFunction


class HierarchicalP2Triangle(ShapeFunction):
    r"""Hierarchical :math:`P_2` basis on the reference triangle (3 vertex hats + 3 edge bubbles).

    In barycentric coordinates :math:`\lambda_0 = 1 - \xi_0 - \xi_1`, :math:`\lambda_1 = \xi_0`,
    :math:`\lambda_2 = \xi_1`. Non-nodal at the P2 DOFs (see the module docstring), so it is used
    only through a ``nodal=False`` element that folds :math:`C = V^{-1}` into the tabulation.
    """

    def __init__(self) -> None:
        """Initialise on the reference triangle."""
        super().__init__(TRIANGLE)

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the six hierarchical basis functions.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, 2)``.

        Returns:
            torch.Tensor: Basis values, shape ``(N_e, N_q, 6)``, ordered ``[v0, v1, v2, e01, e12,
            e20]`` -- the three linear vertex hats :math:`\lambda_i` then the three edge bubbles
            :math:`4\lambda_i\lambda_j`.
        """
        l0 = 1.0 - xi[..., 0] - xi[..., 1]
        l1 = xi[..., 0]
        l2 = xi[..., 1]
        return torch.stack(
            [
                l0,
                l1,
                l2,
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ],
            dim=-1,
        )
