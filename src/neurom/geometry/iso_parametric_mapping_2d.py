import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction


class IsoparametricMapping2D(nn.Module):
    """Class encapsulating mapping from physical to reference coordinates"""

    def __init__(self, shape_function: ShapeFunction):
        super().__init__()
        self.sf = shape_function

    def map(self, xi, x_nodes):
        """
        Maps reference coordinate to physical position based on the elements positions

        Args:
            xi: The reference coordinate (N_e, N_q, dim)
            x_nodes: The nodal points (N_e, N_nodes, dim)

        Returns:
            The positions interpolated in the physical space: (N_e, N_q, dim)
        """
        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi)
        # Sum along N_nodes index
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes)
        return torch.einsum("en...,eqn...->eq...", x_nodes, N)

    def inverse_map(self, x, x_nodes):
        """
        Maps physical position to reference coordinate for linear simplex elements.

        Args:
            x:        (N_e, N_q, dim)      physical coordinates
            x_nodes:  (N_e, N_nodes, dim)  nodal coordinates

        Returns:
            xi:       (N_e, N_q, dim)      reference coordinates

        Note:
            This linear mapping only works for linear shape functions and segment element.
        """
        # Get nodes for convenience (N_e, dim)
        a = x_nodes[:, 0, :]
        b = x_nodes[:, 1, :]
        c = x_nodes[:, 2, :]

        # Get line vectors (N_e, dim)
        ab = b - a
        ac = c - a

        # Affine part (N_e, N_q, dim)
        diff = x - a.unsqueeze(1)

        # Get determinant
        det_J = self.det_jacobian(x_nodes=x_nodes)

        # Inverse matrix (N_e,dim,dim)
        J_inv = torch.stack([ac[:, 1], -ab[:, 1], -ac[:, 0], ab[:, 0]], dim=1).view(
            -1, 2, 2
        ) / det_J.unsqueeze(-1)

        # (N_e, N_q, dim)
        xi = torch.einsum("eql,elk->eqk", diff, J_inv)
        return xi

    def det_jacobian(self, x_nodes):
        """
        Determinant of transformation :math: F(x) = \\xi

        Args:
            x_nodes: The nodal points in physical space (N_e, N_nodes, dim)

        Returns:
            The size of the element.
        """
        # Get nodes for convenience (N_e, dim)
        a = x_nodes[:, 0, :]
        b = x_nodes[:, 1, :]
        c = x_nodes[:, 2, :]

        # Get line vectors (N_e, dim)
        ab = b - a
        ac = c - a

        # Compute determinant (N_e, 1)
        det_J = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        return det_J.unsqueeze(-1)
