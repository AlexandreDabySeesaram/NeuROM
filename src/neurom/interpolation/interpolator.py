from dataclasses import dataclass
import torch
import torch.nn as nn

from neurom.fields.trainable_field import TrainableField
from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_interpolator import QuadratureInterpolator
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)


class Interpolator(nn.Module):
    """Class encapsulating interpolation of a function on a mesh.

    The class provides methods to interpolate a function at quadrature points.
    """

    def __init__(self, mesh, field, sf, quad, mapping):
        super().__init__()
        self.mesh = mesh
        self.field = field
        self.sf = sf
        self.quad = quad

        self.mapping = mapping
        self.field_interp = FieldInterpolator(sf, self.field)
        self.quad_inter = QuadratureInterpolator(self.mesh, self.quad, self.mapping)
        self.quad_pos = self.quad_inter.interpolate()

    def measure(self):
        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.det_jacobian(self.mesh.nodes_positions.at_elements())
        measure = torch.abs(dx) * w
        return measure

    def interpolate(self):
        """
        Interpolate quadrature positions again in case the field is trainable
        """
        if isinstance(self.mesh.nodes_positions, TrainableField):
            self.quad_pos = self.quad_inter.interpolate()

        # Interpolate function
        u_q = self.field_interp.at_reference(self.quad_pos.xi_back)

        # Compute weighted measure
        measure = self.measure()

        # Return everything needed to integrate function
        return QuadratureInterpolationResult(
            x=self.quad_pos.x_phys, u=u_q, measure=measure
        )
