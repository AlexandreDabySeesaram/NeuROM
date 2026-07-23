"""Assembly of interpolated field quantities at quadrature points."""

import torch
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
        context (QuadratureContext): The QuadratureContext with the positions of the quadrature points in physical and reference coordinates.
        sf (ShapeFunction): The ShapeFunction to perform the interpolation.
        field (FieldBase): The FieldBase to interpolate.
        active (bool): Whether ``IntegrationDomain.interpolate_all`` evaluates this assembly. Defaults to ``True``.
    Attributes:
        context (QuadratureContext): The QuadratureContext with the positions of the quadrature points in physical and reference coordinates.
        sf (ShapeFunction): The ShapeFunction to perform the interpolation.
        field (FieldBase): The FieldBase to interpolate.
        active (torch.nn.parameter.Buffer): Bool buffer; when ``False`` the domain skips this assembly. Set in place via :meth:`activate`. Round-trips through ``state_dict``.
        _field_interpolator (FieldInterpolator): The FieldInterpolator used to interpolate the ``field`` with the given shape function ``sf``.
    """

    def __init__(
        self,
        context: QuadratureContext,
        sf: ShapeFunction,
        field: FieldBase,
        active: bool = True,
    ):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)
        # Whether interpolate_all should evaluate this assembly. A registered
        # buffer so it round-trips through state_dict. Monotone for PGD modes
        # (activated, never deactivated); see the single-IntegrationDomain spec.
        self.register_buffer("active", torch.tensor(bool(active)))

    def activate(self) -> None:
        """Mark this assembly for interpolation (in place; keeps buffer identity)."""
        self.active.fill_(True)

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
