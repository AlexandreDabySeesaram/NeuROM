import torch
import torch.nn as nn

from neurom.fields.trainable_field import TrainableField
from neurom.meshes.mesh import Mesh
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_interpolator import QuadratureInterpolator
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)


class Interpolator(nn.Module):
    def __init__(
        self,
        mesh: Mesh,
        quad: QuadratureRule,
        mapping,
        field_interpolators: list[FieldInterpolator],
    ):
        super().__init__()
        self.mesh = mesh
        self.quad = quad
        self.mapping = mapping
        self.quad_inter = QuadratureInterpolator(self.mesh, self.quad, self.mapping)
        self.field_interpolators = field_interpolators

        self.quad_pos = self.quad_inter.interpolate()

    def measure(self):
        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.det_jacobian(self.mesh.nodes_positions.at_elements())
        m = torch.abs(dx) * w
        return m

    def interpolate(self, field_layout):
        if isinstance(self.quad_inter.mesh.nodes_positions, TrainableField):
            self.quad_pos = self.quad_inter.interpolate()

        # Compute weighted measure
        measure = self.measure()

        # Interpolate function
        for field_interp in self.field_interpolators:
            u_q = field_interp.at_reference(self.quad_pos.xi_back)
            result = QuadratureInterpolationResult(
                x=self.quad_pos.x_phys, u=u_q, measure=measure
            )

            field_layout.update(field_interp.field, result)
