import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase


class FieldInterpolator(nn.Module):
    """Class encapsulating interpolation of a Field

    The class provides a way to interpolate a FieldBase based on a ShapeFunction
    """

    def __init__(self, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.sf = sf
        self.field = field

    def at_reference(self, xi: torch.tensor):
        N = self.sf.N(xi)
        return torch.einsum("en...,eqn...->eq...", self.field.at_elements(), N)
