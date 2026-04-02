import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.fields.field_base import FieldBase
from neurom.meshes.mesh import Mesh


class PointWiseInterpolator(nn.Module):
    """Class encapsulating interpolation of a field at arbitrary points"""

    def __init__(self, mesh: Mesh, sf: ShapeFunction, field: FieldBase, mapping):
        super().__init__()
        self.mesh = mesh
        self.sf = sf
        self.field = field
        self._mapping = mapping

    def at_position(self, x: torch.Tensor):
        element_ids = self.mesh.elements_at(x)

        # Get connectivity for those elements
        element_nodes_ids = self.mesh.topology.connectivity[element_ids, :]

        # (N_e, N_q, dim)
        xi = self._mapping.inverse_map(x)
        N = self.sf.N(xi)
        u = torch.einsum(
            "en...,eqn...->eq...", self.field.full_values()[element_nodes_ids], N
        )

        return u
