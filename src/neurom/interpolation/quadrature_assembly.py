"""Assembly of interpolated field quantities at quadrature points."""

import torch.nn as nn

from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.fields.field_base import FieldBase
from neurom.shape_functions.shape_function import ShapeFunction

from neurom.interpolation.quadrature_assembly_result import (
    QuadratureAssemblyResult,
)
from neurom.samplings import QuadratureSampling


class QuadratureAssembly(nn.Module):
    """Assembles the field interpolation at quadrature points.

    Combines a ``QuadratureContext`` (which holds geometric information such as
    physical and reference positions and the integration measure) with a
    ``ShapeFunction`` and a ``FieldBase`` to produce a
    ``QuadratureAssemblyResult`` ready for numerical integration.

    Args:
        context (QuadratureContext): Provides the quadrature positions in
            physical and reference coordinates as well as the integration
            measure.
        sf (ShapeFunction): The shape function used to perform the
            interpolation.
        field (FieldBase): The field whose nodal values are interpolated at
            the quadrature points.

    Attributes:
        context (QuadratureContext): Provides the quadrature positions in
            physical and reference coordinates as well as the integration
            measure.
        sf (ShapeFunction): The shape function used to perform the
            interpolation.
        field (FieldBase): The field whose nodal values are interpolated at
            the quadrature points.
        _field_interpolator (FieldInterpolator): Internal interpolator that
            evaluates ``field`` using ``sf`` at reference coordinates.
    """

    def __init__(self, context: QuadratureContext, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)

    def interpolate(self) -> QuadratureAssemblyResult:
        """Interpolate the field at all quadrature points.

        Retrieves the integration measure and quadrature positions from
        ``self.context``, evaluates ``self.field`` at the back-mapped reference
        coordinates, and bundles everything into a ``QuadratureAssemblyResult``.

        Returns:
            QuadratureAssemblyResult: Contains the physical positions ``x``,
            the interpolated field values ``u``, and the integration measure,
            all as ``QuadratureSampling`` objects of shape
            ``(N_e, N_q, *)``.
        """
        # Get measure and quadrature positions from context
        measure = self.context.measure
        quad_pos = self.context.interpolate

        # Interpolate field
        u_q = QuadratureSampling(
            self._field_interpolator.at_reference(quad_pos.xi_back.values)
        )

        # Assemble the result
        result = QuadratureAssemblyResult(x=quad_pos.x_phys, u=u_q, measure=measure)

        return result
