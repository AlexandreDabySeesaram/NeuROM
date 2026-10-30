"""Linear shape functions on the reference bar."""

import torch

from neurom.reference_cells import BAR
from neurom.shape_functions.shape_function import ShapeFunction


class LinearBar(ShapeFunction):
    """Linear (P1) shape function defined on the 1-D reference bar ``[-1, 1]``.

    The two nodal basis functions are:

    .. math::

        N_1(\\xi) = \\tfrac{1}{2}(1 - \\xi), \\quad
        N_2(\\xi) = \\tfrac{1}{2}(1 + \\xi).
    """

    def __init__(self):
        """Initialise using the reference bar."""
        super().__init__(BAR)

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

    def grad_N(self, xi):
        r"""Constant reference gradient :math:`\mathrm dN/\mathrm d\xi = [-\tfrac12, \tfrac12]`.

        Overrides the autograd default with the closed form (P1 is affine, so the gradient does not
        depend on :math:`\xi`), broadcast to the input batch.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, 1)``.

        Returns:
            torch.Tensor: :math:`\partial N/\partial\xi`, shape ``(N_e, N_q, 2, 1)``.
        """
        g = torch.tensor([[-0.5], [0.5]], dtype=xi.dtype, device=xi.device)  # (2, 1)
        return g.expand(*xi.shape[:-1], 2, 1)
