import torch
import torch.nn as nn

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
        context (QuadratureContext): The QuadratureContext with the positions of the quadrature points in physical and reference coordinates.
        sf (ShapeFunction): The ShapeFunction to perform the interpolation.
        field (FieldBase): The FieldBase to interpolate.
    Attributes:
        context (QuadratureContext): The QuadratureContext with the positions of the quadrature points in physical and reference coordinates.
        sf (ShapeFunction): The ShapeFunction to perform the interpolation.
        field (FieldBase): The FieldBase to interpolate.
        _field_interpolator (FieldInterpolator): The FieldInterpolator used to interpolate the ``field`` with the given shape function ``sf``.
    """

    def __init__(self, context: QuadratureContext, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)

    def interpolate(self) -> QuadratureAssemblyResult:
        """The main interpolation method

        Interpolate the field and associates it with the quadrature positions at which it is interpolated and the measure of the element and quadrature points.

        Returns:
            (QuadratureAssemblyResult) which encapsulates the positions and the interpolated field at the given positions as well as the measure.
        """
        # Get measure and quadrature positions from context
        measure = self.context.measure
        quad_pos = self.context.interpolate

        # Interpolate field
        u_q = self._field_interpolator.at_reference(quad_pos.xi_back)

        # Assemble the result
        result = QuadratureAssemblyResult(x=quad_pos.x_phys, u=u_q, measure=measure)

        return result
