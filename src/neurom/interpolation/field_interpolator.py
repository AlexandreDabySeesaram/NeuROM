"""Interpolation of a field at reference coordinates using shape functions."""

from dataclasses import replace

import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase
from neurom.samplings import Sampling


class FieldInterpolator(nn.Module):
    """Interpolates a ``FieldBase`` at reference coordinates using a ``ShapeFunction``.

    Wraps a field and its associated shape function to evaluate the field at
    arbitrary reference-element coordinates via a finite-element interpolation.

    Args:
        sf (ShapeFunction): The shape function used to build the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.

    Attributes:
        sf (ShapeFunction): The shape function used to build the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.
    """

    def __init__(self, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.sf = sf
        self.field = field

    def at_reference(self, xi: Sampling, u_elem: torch.Tensor | None = None) -> Sampling:
        """Interpolate the field at reference-element coordinates.

        Evaluates the shape functions at ``xi`` and contracts them with the element-wise DOFs. The
        contraction (a reduction over the DOF axis) is a direct einsum on ``xi.values``, rewrapped
        with the input's subtype/batch via :func:`dataclasses.replace` -- the same Sampling-native
        pattern as :meth:`neurom.geometry.AffineSimplexMapping.map` (which contracts geometry nodes
        rather than field DOFs). Not an :func:`~neurom.apply.apply` op: ``apply`` broadcasts, this
        reduces.

        Args:
            xi (Sampling): Reference coordinates, ``f_shape == (dim,)`` and a leading element axis
                (e.g. a ``QuadratureSampling`` of batch ``(N_e, N_q)``).
            u_elem (torch.Tensor, optional): Reference DOFs gathered per element, tensor of shape
                ``(N_e, N_dofs, *field_dim)``. Defaults to ``self.field.at_elements()``.

        Returns:
            Sampling: Interpolated field values, same subtype/batch as ``xi``,
            ``f_shape == (*field_dim,)``.
        """
        if u_elem is None:
            u_elem = self.field.at_elements()
        N = self.sf.N(xi.values)
        return replace(xi, values=torch.einsum("en...,eqn...->eq...", u_elem, N))
