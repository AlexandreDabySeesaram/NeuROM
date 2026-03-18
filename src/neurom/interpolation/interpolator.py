import torch
import torch.nn as nn

from neurom.fields.trainable_field import TrainableField
from neurom.field_layout import FieldLayout
from neurom.meshes.mesh import Mesh
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_interpolator import QuadratureInterpolator
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)


class Interpolator(nn.Module):
    """General interpolator

    This class encapsulates interpolation workflow: it interpolates positions at quadrature points, interpolate all provided fields and compute associated measure (determinant of Jacobian times quadrature points).

    The result of the interpolation are written to a FieldLayout object.

    Args:
        mesh (Mesh): The mesh on which the interpolation is performed.
        quad (QuadratureRule): The actual quadrature rule to consider for integration.
        mapping: The mapping to use from reference to physical coordinates.
        field_interpolators (list[FieldInterpolator]): The list of FieldInterpolator. Represents all the fields we wish to interpolate.
    Attributes:
        mesh (Mesh): The mesh on which the interpolation is performed.
        quad (QuadratureRule): The actual quadrature rule to consider for integration.
        mapping: The mapping to use from reference to physical coordinates.
        field_interpolators (list[FieldInterpolator]): The list of FieldInterpolator. Represents all the fields we wish to interpolate.
        quad_inter (QuadratureInterpolator): The interpolator strategy for positions at quadrature points.
        quad_pos (QuadratureInterpolator): The initial result of interpolation of positions. Provides the reference coordinates on which interpolation of ``.field_interpolators`` will be done.
    """

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
        self.field_interpolators = field_interpolators
        self.quad_inter = QuadratureInterpolator(self.mesh, self.quad, self.mapping)
        self.quad_pos = self.quad_inter.interpolate()

    def measure(self) -> torch.Tensor:
        """Helper method to compute the 'measure'

        Returns:
            The product of the determinant of the jacobian from physical to reference coordinates mapping times the quadrature weights. Tensor of shape (N_e, N_q).
        """
        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.det_jacobian(self.mesh.nodes_positions.at_elements())
        m = torch.abs(dx) * w
        return m

    def interpolate(self, field_layout: FieldLayout) -> None:
        """The main interpolation method

        If the self.mesh.nodes_positions are a TrainableField, reperform interpolation of positions.
        Then, compute the measure().
        Finally, iterate for each field in self.field_interpolators and compute interpolation at reference coordinate (self.quad_pos.xi_back) and update() the result in the field_layout.

        Args:
            field_layout (FieldLayout) : The field layout on which we will write the result of the interpolation.
        """
        if isinstance(self.mesh.nodes_positions, TrainableField):
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
