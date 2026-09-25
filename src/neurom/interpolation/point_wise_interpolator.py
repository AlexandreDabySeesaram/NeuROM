"""Point-wise interpolation of fields at arbitrary physical positions."""

import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase
from neurom.meshes.mesh import Mesh


class PointWiseInterpolator(nn.Module):
    """Interpolates a field at arbitrary physical positions.

    For each query point this interpolator locates the containing element,
    maps the point back to reference coordinates via the inverse mapping,
    evaluates the shape functions, and applies a finite-element interpolation
    using the field's nodal values.

    Args:
        mesh (Mesh): The mesh on which the field is defined.
        sf (ShapeFunction): The shape function used for the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.
        mapping: The geometric mapping that provides ``inverse_map_at``,
            translating physical positions to reference coordinates for
            specified elements.

    Attributes:
        mesh (Mesh): The mesh on which the field is defined.
        sf (ShapeFunction): The shape function used for the interpolation.
        field (FieldBase): The field whose nodal values are interpolated.
        _mapping: The geometric mapping providing the inverse map from
            physical to reference coordinates.
    """

    def __init__(self, mesh: Mesh, sf: ShapeFunction, field: FieldBase, mapping):
        super().__init__()
        self.mesh = mesh
        self.sf = sf
        self.field = field
        self._mapping = mapping

    def at_position(self, x: torch.Tensor):
        """Interpolate the field at the given physical positions.

        For each point in ``x`` the method:

        1. Finds the containing element using ``self.mesh.elements_at``.
        2. Retrieves the element node indices from the mesh connectivity.
        3. Computes reference coordinates via ``self._mapping.inverse_map_at``.
        4. Evaluates shape functions and contracts with the nodal field values.

        Args:
            x (torch.Tensor): Physical query positions, tensor of shape
                ``(N_pts, dim)`` — one query point per row.

        Returns:
            torch.Tensor: Interpolated field values at each query point,
            tensor of shape ``(N_pts, field_dim)``.

        Raises:
            ValueError: If ``x`` does not have shape ``(N_pts, dim)``.
        """
        # Guard the rank explicitly: a tensor of the wrong rank does NOT fail
        # downstream, it broadcasts inside `inverse_map_at` into a
        # point-by-element cross product, which the shape function then slices
        # back down to the *correct output shape* with wrong values. Silent
        # numerical corruption; caught here instead. `elements_at` runs the
        # same check, see there for the full story.
        if x.ndim != 2 or x.shape[-1] != self.mesh.dim:
            raise ValueError(
                f"at_position expects x of shape (N_pts, dim) with "
                f"dim={self.mesh.dim}, got {tuple(x.shape)}. Reshape a flat "
                f"list of points with x.reshape(-1, {self.mesh.dim})."
            )

        element_ids = self.mesh.elements_at(x)
        # Get connectivity for those elements
        element_nodes_ids = self.mesh.connectivity.element_connectivity[element_ids, :]

        # `inverse_map_at` and the shape functions work on the quadrature
        # layout (N_e, N_q, dim); a point-wise query is that layout with a
        # single "quadrature point" per element, dropped again on the way out.
        # (N_pts, 1, dim)
        xi = self._mapping.inverse_map_at(x.unsqueeze(1), element_ids)
        N = self.sf.N(xi)
        u = torch.einsum(
            "en...,eqn...->eq...", self.field.full_values()[element_nodes_ids], N
        )

        # (N_pts, 1, field_dim) -> (N_pts, field_dim)
        return u.squeeze(1)
