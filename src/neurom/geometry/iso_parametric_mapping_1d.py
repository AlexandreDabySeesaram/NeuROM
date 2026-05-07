import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.meshes.mesh import Mesh


class IsoparametricMapping1D(nn.Module):
    """Class encapsulating mapping from physical to reference coordinates"""

    def __init__(self, shape_function: ShapeFunction, mesh: Mesh):
        super().__init__()
        self.sf = shape_function
        self._mesh = mesh
        self.x_nodes = self._mesh.nodes_positions.at_elements()

    def map(self, xi):
        """
        Maps reference coordinate to physical position based on the mesh elements positions

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
        # Center point per element
        # (N_e, dim)
        x_half = 0.5 * (self.x_nodes[:, 1, :] + self.x_nodes[:, 0, :])

        # Inverse mapping
        # (N_e, dim)
        det_F_inv = 1.0 / self.det_jacobian

        # Offset positions
        # (N_e, N_q, dim)
        offset = x - x_half.unsqueeze(1)

        # Compute reference position
        xi = offset * det_F_inv.unsqueeze(1)

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
        # Center point per element
        # (N_e, dim)
        x_nodes = self.x_nodes[element_ids]
        x_half = 0.5 * (x_nodes[:, 1, :] + x_nodes[:, 0, :])

        # Inverse mapping
        # (N_e, dim)
        det_F_inv = 1.0 / self.det_jacobian[element_ids]

        # Offset positions
        # (N_e, N_q, dim)
        offset = x - x_half.unsqueeze(1)

        # Compute reference position
        xi = offset * det_F_inv.unsqueeze(1)

        return xi

    @property
    def det_jacobian(self):
        """
        Determinant of transformation :math: F(x) = \\xi

        Returns:
            The size of the element.
        """
        return 0.5 * (self.x_nodes[:, 1, :] - self.x_nodes[:, 0, :])

    def update(self):
        self.x_nodes = self._mesh.nodes_positions.at_elements()
