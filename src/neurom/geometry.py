import torch
import torch.nn as nn

from hidenn_playground.shape_functions import ShapeFunction


class IsoparametricMapping1D(nn.Module):
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
        # Center pointer per element
        # (N_e, dim)
        x_half = 0.5 * (x_nodes[:, 1, :] + x_nodes[:, 0, :])

        # Inverse mapping
        # (N_e, dim)
        J_inv = 2.0 / self.element_size(x_nodes)

        # Offset positions
        # (N_e, N_q, dim)
        offset = x - x_half.unsqueeze(1)

        # Compute reference position
        xi = offset * J_inv.unsqueeze(1)

        return xi

    def element_size(self, x_nodes):
        """
        Computes the size of the physical elements.

        Args:
            x_nodes: The nodal points in physical space (N_e, N_nodes, dim)

        Returns:
            The size of the element.
        """
        return x_nodes[:, 1, :] - x_nodes[:, 0, :]
