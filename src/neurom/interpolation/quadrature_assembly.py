import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.fields.field_base import FieldBase
from neurom.shape_functions.shape_function import ShapeFunction

from neurom.interpolation.quadrature_assembly_result import (
    QuadratureAssemblyResult,
)


class QuadratureAssembly(nn.Module):
    """Assemble the interpolation at quadrature points

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

    def __init__(self, context: QuadratureContext, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)

    def interpolate(self) -> QuadratureAssemblyResult:
        """The main interpolation method
        )
                If the self.mesh.nodes_positions are a TrainableField, reperform interpolation of positions.
                Finally, iterate for each field in self.field_interpolators and compute interpolation at reference coordinate (self.quad_pos.xi_back) and update() the result in the field_layout.

                Args:
        """
        # Get measure and quadrature positions from context
        measure = self.context.measure
        quad_pos = self.context.interpolate

        # Interpolate field
        u_q = self._field_interpolator.at_reference(quad_pos.xi_back)

        # Assemble the result
        result = QuadratureAssemblyResult(x=quad_pos.x_phys, u=u_q, measure=measure)

        return result
