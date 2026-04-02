import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.meshes.mesh import Mesh


class IsoparametricMapping2D(nn.Module):
    """Class encapsulating mapping from physical to reference coordinates"""

    def __init__(self, shape_function: ShapeFunction, mesh: Mesh):
        super().__init__()
        self.sf = shape_function
        self.mesh = mesh
        self.x_nodes = self.mesh.nodes_positions.at_elements()
        self._J_inv_cache = None
        self._det_J_cache = None

    def _compute_J_inv(self):
        # Get nodes for convenience (N_e, dim)
        a = self.x_nodes[:, 0, :]
        b = self.x_nodes[:, 1, :]
        c = self.x_nodes[:, 2, :]

        # Get line vectors (N_e, dim)
        ab = b - a
        ac = c - a

        # Get determinant
        det_J = self.det_jacobian

        # Inverse matrix (N_e,dim,dim)
        J_inv = torch.stack([ac[:, 1], -ab[:, 1], -ac[:, 0], ab[:, 0]], dim=1).view(
            -1, 2, 2
        ) / det_J.unsqueeze(-1)

        # Return it as (N_e,1,dim,dim) i.e. with N_q=1
        return J_inv

    def _compute_det_J(self):
        """
        Determinant of transformation :math: F(x) = \\xi

        Returns:
            The size of the element.
        """
        # Get nodes for convenience (N_e, dim)
        a = self.x_nodes[:, 0, :]
        b = self.x_nodes[:, 1, :]
        c = self.x_nodes[:, 2, :]

        # Get line vectors (N_e, dim)
        ab = b - a
        ac = c - a

        # Compute determinant and return it as (N_e, 1) i.e. with N_q=1
        det_J = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        return det_J.unsqueeze(-1)

    @property
    def J_inv(self):
        from neurom.fields.trainable_field import TrainableField

        if isinstance(self.mesh.nodes_positions, TrainableField):
            return self._compute_J_inv()
        if self._J_inv_cache is None:
            self._J_inv_cache = self._compute_J_inv()

        return self._J_inv_cache

    @property
    def det_jacobian(self):
        from neurom.fields.trainable_field import TrainableField

        if isinstance(self.mesh.nodes_positions, TrainableField):
            return self._compute_det_J()
        if self._det_J_cache is None:
            self._det_J_cache = self._compute_det_J()

        return self._det_J_cache

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
            This linear mapping only works for linear shape functions and segment element.
        """

        # Recover x shape
        N_e_x, N_q, _ = x.shape

        # Shape 'a' into (N_e, N_q, dim)
        a = self.x_nodes[:, 0, :].unsqueeze(1).expand(-1, N_q, -1)

        # Do we exactly have same number of elements in ``x`` than in mesh?
        # - Yes - Easy lookup
        if N_e_x == self.mesh.n_elements:
            xi = torch.einsum("eql,elk->eqk", x - a, self.J_inv)
            return xi

        # - No - Find elements to which x belongs to for all points in N_q
        # Restrict interpolation on those elements
        elements_ids = self.mesh.elements_at(x)
        xi = torch.einsum(
            "eql,eqlk->eqk",
            x - a[elements_ids],
            self.J_inv[elements_ids].unsqueeze(1).expand(-1, N_q, -1, -1),
        )
        return xi
