"""Constant (DG0) shape function: one cell-associated DOF per element."""

import torch

from neurom.shape_functions.shape_function import ShapeFunction


class Constant(ShapeFunction):
    r"""Piecewise-constant (DG0) element: a single DOF, the cell value.

    The reference basis is :math:`\hat H_1 \equiv 1`, so a field interpolates to a
    constant per element.  The physical DOF is the cell average, whose transformation
    is the identity (:math:`M_e = 1`).

    Used as the reference basis of :func:`neurom.elements.DG0`, which declares the
    single cell DOF; a ``FunctionSpace`` then gives one unshared DOF per element.
    Works on any reference element (bar, triangle).
    """

    def __init__(self, reference_cell):
        """Initialise on a given reference element.

        Args:
            reference_cell (ReferenceCell): The reference element (e.g.
                bar in 1-D, triangle in 2-D).
        """
        super().__init__(reference_cell)

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate the single constant basis function.

        Args:
            xi (torch.Tensor): Reference coordinates of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Ones of shape ``(N_e, N_q, 1)``.
        """
        return xi.new_ones((*xi.shape[:-1], 1))
