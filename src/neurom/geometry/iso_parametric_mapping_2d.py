import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.meshes.mesh import Mesh


class IsoparametricMapping2D(nn.Module):
    """Class encapsulating mapping from physical to reference coordinates"""

    def __init__(self, shape_function: ShapeFunction, mesh: Mesh):
        super().__init__()
        self.sf = shape_function
        self._mesh = mesh
        self.x_nodes = self._mesh.nodes_positions.at_elements()
        self._compute_J_inv()

    def _compute_J_inv(self):
        # Get nodes for convenience (N_e, dim)
        a = self.x_nodes[:, 0, :]
        b = self.x_nodes[:, 1, :]
        c = self.x_nodes[:, 2, :]

        # Get line vectors (N_e, dim)
        ab = b - a
        ac = c - a

        # Compute determinant and holds it as (N_e, 1) i.e. with N_q=1
        self._det_J = (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).unsqueeze(-1)

        # Inverse matrix (N_e,dim,dim)
        self._J_inv = torch.stack(
            [ac[:, 1], -ab[:, 1], -ac[:, 0], ab[:, 0]], dim=1
        ).view(-1, 2, 2) / self.det_jacobian.unsqueeze(-1)

    @property
    def J_inv(self):
        return self._J_inv

    @property
    def det_jacobian(self):
        return self._det_J

    def map(self, xi):
        """
        Maps reference coordinate to physical position based on the elements positions

        Args:
            xi: The reference coordinate (N_e, N_q, dim)

        Returns:
            The positions interpolated in the physical space: (N_e, N_q, dim)
        """
        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi)
        # Sum along N_nodes index
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes)
        return torch.einsum("en...,eqn...->eq...", self.x_nodes, N)

    def inverse_map(self, x):
        """
        Maps physical position to reference coordinate for linear simplex elements.

        Args:
            x:        (N_e, N_q, dim)      physical coordinates

        Returns:
            xi:       (N_e, N_q, dim)      reference coordinates

        Note:
            This linear mapping only works for linear shape functions and bar element.
        """

        # Recover x shape
        _, N_q, _ = x.shape

        # Shape 'a' into (N_e, N_q, dim)
        a = self.x_nodes[:, 0, :].unsqueeze(1).expand(-1, N_q, -1)

        xi = torch.einsum("eql,elk->eqk", x - a, self.J_inv)

        return xi

    def inverse_map_at(self, x, element_ids):
        """
        Maps physical position to reference coordinate for linear simplex elements.

        Args:
            x:        (N_e, N_q, dim)      physical coordinates
            element_ids:        (N_e,)     indices of elements to use to compute the inverse map

        Returns:
            xi:       (N_e, N_q, dim)      reference coordinates

        Note:
            This linear mapping only works for linear shape functions and bar element.
        """

        # Recover x shape
        _, N_q, _ = x.shape

        # Shape 'a' into (N_e, N_q, dim)
        a = self.x_nodes[:, 0, :].unsqueeze(1).expand(-1, N_q, -1)

        # Restrict interpolation on given elements
        xi = torch.einsum(
            "eql,eqlk->eqk",
            x - a[element_ids],
            self.J_inv[element_ids].unsqueeze(1).expand(-1, N_q, -1, -1),
        )
        return xi

    def update(self):
        self.x_nodes = self._mesh.nodes_positions.at_elements()
        self._compute_J_inv()
