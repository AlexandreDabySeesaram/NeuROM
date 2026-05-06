import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase


class FieldInterpolator(nn.Module):
    """Class encapsulating interpolation of a Field

    The class provides a way to interpolate a FieldBase based on a ShapeFunction

    Args:
        sf (ShapeFunction): The shape function to use for interpolation.
        field (FieldBase): The field to interpolate.

    Attributes:
        sf (ShapeFunction): The shape function to use for interpolation.
        field (FieldBase): The field to interpolate.
    """

    def __init__(self, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.sf = sf
        self.field = field

    def at_reference(self, xi: torch.tensor):
        """Interpolate field at reference position

        Args:
            xi (torch.Tensor): The reference coordinate, tensor of shape (N_e, N_q, dim)

        Returns:
            The interpolated field interpolated, tensor of shape (N_e, N_q, dim)
        """
        N = self.sf.N(xi)
        return torch.einsum("en...,eqn...->eq...", self.field.at_elements(), N)
