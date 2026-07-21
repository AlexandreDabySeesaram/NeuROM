"""Linear shape functions on the reference bar."""

import torch

from neurom.reference_elements.bar import Bar
from neurom.shape_functions.shape_function import ShapeFunction


class LinearBar(ShapeFunction):
    """Linear (P1) shape function defined on the 1-D reference bar ``[-1, 1]``.

    The two nodal basis functions are:

    .. math::

        N_1(\\xi) = \\tfrac{1}{2}(1 - \\xi), \\quad
        N_2(\\xi) = \\tfrac{1}{2}(1 + \\xi).
    """

    def __init__(self):
        """Initialise using the standard ``Bar`` reference element."""
        super().__init__(Bar())

    def N(self, xi):
        """Evaluate the two linear shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, 2)``.
        """
        xi0 = xi[..., 0]

        return torch.stack(
            [
                -0.5 * (xi0 - 1.0),
                0.5 * (xi0 + 1.0),
            ],
            dim=-1,
        )
