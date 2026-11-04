"""Interpolation of a field at reference coordinates using shape functions."""

import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase


class FieldInterpolator(nn.Module):
    """Interpolates a ``FieldBase`` at reference coordinates using a ``ShapeFunction``.

    Wraps a field and its associated shape function to evaluate the field at
    arbitrary reference-element coordinates via a finite-element interpolation.

    Args:
        sf (ShapeFunction): The shape function used to build the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.

    Attributes:
        sf (ShapeFunction): The shape function used to build the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.
    """

    def __init__(self, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.sf = sf
        self.field = field

    def at_reference(self, xi: torch.Tensor, u_elem: torch.Tensor | None = None):
        """Interpolate the field at reference-element coordinates.

        Evaluates the shape functions at ``xi`` and contracts them with the
        element-wise DOFs via an Einstein summation.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim)``.
            u_elem (torch.Tensor, optional): Reference DOFs gathered per
                element, tensor of shape ``(N_e, N_dofs, field_dim)``.
                Defaults to ``self.field.at_elements()``.

        Returns:
            torch.Tensor: Interpolated field values, tensor of shape
            ``(N_e, N_q, field_dim)``.
        """
        if u_elem is None:
            u_elem = self.field.at_elements()
        N = self.sf.N(xi)
        return torch.einsum("en...,eqn...->eq...", u_elem, N)
