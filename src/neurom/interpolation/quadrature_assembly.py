"""Assembly of interpolated field quantities at quadrature points."""

import torch.nn as nn

from neurom.interpolation.field_interpolator import FieldInterpolator
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.fields.field_base import FieldBase
from neurom.dof_transformations import Transformation
from neurom.dof_transformations.reference_to_physical import reference_to_physical_at

from neurom.interpolation.quadrature_assembly_result import (
    QuadratureAssemblyResult,
)


class QuadratureAssembly(nn.Module):
    """Assembles a field's interpolation at the quadrature points.

    Combines a :class:`QuadratureContext` (physical/reference positions and the
    integration measure) with a :class:`~neurom.fields.FieldBase` to produce a
    :class:`QuadratureAssemblyResult` ready for numerical integration. The element basis
    and the physical-to-reference DOF transformation are both derived from the field's
    space (``field.space.element``) and the context's mapping.

    Args:
        context (QuadratureContext): Holds the quadrature points in physical and reference
            coordinates, the integration measure, and the geometric mapping.
        field (FieldBase): The field to interpolate; its space supplies the element.

    Attributes:
        context (QuadratureContext): The quadrature context.
        field (FieldBase): The interpolated field.
        sf (ShapeFunction): The element's reference basis.
    """

    def __init__(self, context: QuadratureContext, field: FieldBase):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = field.space.element.reference_basis
        self._field_interpolator = FieldInterpolator(self.sf, self.field)
        # Physical element DOFs -> reference coefficients (identity for nodal Lagrange).
        self._dof_transformation = Transformation(field.space.element, context.mapping)

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

        # Physical element DOFs to reference coefficients (identity for nodal elements)
        u_elem = self.field.at_elements()
        u_elem = self._dof_transformation.to_reference(u_elem)

        # Interpolate field (returns a Sampling, same batch as the quadrature positions)
        u_q = self._field_interpolator.at_reference(quad_pos.xi_back, u_elem)

        # Push the reference value forward to physical space (identity no-op for scalar/Lagrange).
        u_q = reference_to_physical_at(
            u_q,
            self.context.mapping,
            quad_pos.xi_back,
            self.field.space.element.map_type,
        )

        # Assemble the result
        result = QuadratureAssemblyResult(x=quad_pos.x_phys, u=u_q, measure=measure)

        return result
