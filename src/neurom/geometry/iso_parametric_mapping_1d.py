"""Isoparametric mapping between reference and physical coordinates for 1D elements."""

import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.meshes.mesh import Mesh


class IsoparametricMapping1D(nn.Module):
    """Isoparametric mapping between reference and physical space for 1-D elements.

    Implements the forward map :math:`x = \\sum_n N_n(\\xi)\\, x_n` and
    its analytical inverse for linear bar elements.

    Attributes:
        sf (ShapeFunction): Shape-function object used to evaluate basis
            functions and their derivatives.
        x_nodes (torch.Tensor): Physical node coordinates gathered per
            element, shape ``(N_e, N_nodes, dim)``.
    """

    def __init__(self, shape_function: ShapeFunction, mesh: Mesh):
        """Initialise the mapping from a shape function and a mesh.

        Args:
            shape_function (ShapeFunction): Shape-function object compatible
                with the mesh element type.
            mesh (Mesh): Mesh whose node positions define the physical domain.
        """
        super().__init__()
        self.sf = shape_function
        self._mesh = mesh
        self.x_nodes = self._mesh.nodes_positions.at_elements()

    def map(self, xi):
        """Map reference coordinates to physical positions.

        Evaluates :math:`x = \\sum_n N_n(\\xi)\\, x_n` element-wise.

        Args:
            xi (torch.Tensor): Reference coordinates of shape
                ``(N_e, N_q, dim)``.

        Returns:
            torch.Tensor: Physical positions of shape ``(N_e, N_q, dim)``.
        """
        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi)
        # Sum along N_nodes index
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes)
        return torch.einsum("en...,eqn...->eq...", self.x_nodes, N)

    def inverse_map(self, x):
        """Map physical positions to reference coordinates over all elements.

        Applies the analytic inverse of the linear isoparametric map for
        bar elements: :math:`\\xi = (x - x_{\\text{mid}}) / \\det F`.

        Args:
            x (torch.Tensor): Physical coordinates of shape
                ``(N_e, N_q, dim)``.

        Returns:
            torch.Tensor: Reference coordinates of shape ``(N_e, N_q, dim)``.

        Note:
            This inverse mapping is exact only for linear shape functions on
            bar elements.
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
        """Map physical positions to reference coordinates using a subset of elements.

        Same analytic inverse as :meth:`inverse_map`, but restricts the
        computation to the elements identified by ``element_ids``.

        Args:
            x (torch.Tensor): Physical coordinates of shape
                ``(N_e, N_q, dim)``.
            element_ids (torch.Tensor): Indices of the elements to use,
                shape ``(N_e,)``.

        Returns:
            torch.Tensor: Reference coordinates of shape ``(N_e, N_q, dim)``.

        Note:
            This inverse mapping is exact only for linear shape functions on
            bar elements.
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
        """Jacobian of the reference-to-physical map for each element.

        For a 1-D bar element the Jacobian equals half the element length:
        :math:`F = \\frac{x_1 - x_0}{2}`.

        Returns:
            torch.Tensor: Per-element Jacobian values, shape ``(N_e, dim)``.
        """
        return 0.5 * (self.x_nodes[:, 1, :] - self.x_nodes[:, 0, :])

    def update(self):
        """Refresh cached node positions from the underlying mesh.

        Must be called whenever the mesh node positions change (e.g. after a
        training step that moves the nodes) so that subsequent calls to
        :meth:`map` and :meth:`inverse_map` use the updated geometry.
        """
        self.x_nodes = self._mesh.nodes_positions.at_elements()
