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

    def at_reference(self, xi: torch.Tensor):
        """Interpolate the field at reference-element coordinates.

        Evaluates the shape functions at ``xi`` and contracts them with the
        element-wise nodal values of ``self.field`` via an Einstein summation.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim)``.

        Returns:
            torch.Tensor: Interpolated field values, tensor of shape
            ``(N_e, N_q, field_dim)``.
        """
        N = self.sf.N(xi)
        return torch.einsum("en...,eqn...->eq...", self.field.at_elements(), N)
