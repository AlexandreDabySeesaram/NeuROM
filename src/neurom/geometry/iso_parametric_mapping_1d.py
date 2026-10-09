"""Isoparametric mapping between reference and physical coordinates for 1D elements."""

import torch
import torch.nn as nn

from neurom.meshes.mesh import Mesh
from neurom.math.jacobian import jacobian


class IsoparametricMapping1D(nn.Module):
    """Isoparametric mapping between reference and physical space for 1-D elements.

    Implements the forward map :math:`x = \\sum_n N_n(\\xi)\\, x_n` and its analytical
    inverse for linear bar elements. The geometry basis is taken from the mesh's coordinate
    element (``mesh.coordinates.space.element``), so there is no separate shape function to
    pass in.

    Attributes:
        sf (ShapeFunction): The coordinate element's reference basis.
        x_nodes (torch.Tensor): Physical node coordinates gathered per element, shape
            ``(N_e, N_nodes, dim)``.
    """

    def __init__(self, mesh: Mesh):
        """Initialise the mapping from a mesh.

        Args:
            mesh (Mesh): Mesh whose coordinate field defines the physical domain (and,
                through its space, the geometry basis).
        """
        super().__init__()
        self.sf = mesh.coordinates.space.element.reference_basis
        self._mesh = mesh
        self.x_nodes = self._mesh.coordinates.at_elements()

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

    def jacobian_at(self, xi, element_ids=None):
        """Jacobian of the reference-to-physical map at given reference points.

        Evaluates :math:`J = \\partial x / \\partial \\xi` by differentiating
        :math:`x = \\sum_n N_n(\\xi)\\, x_n` with autograd, so it holds for any
        geometric shape function.  The graph to the node positions is kept,
        so gradients flow to trainable nodes.

        Args:
            xi (torch.Tensor): Reference coordinates of shape
                ``(N_e, N_p, dim)``.
            element_ids (torch.Tensor, optional): Indices of the elements to
                use, shape ``(N_e,)``.  All elements when ``None``.

        Returns:
            torch.Tensor: Jacobian of shape ``(N_e, N_p, dim, dim)``.
        """
        x_nodes = self.x_nodes if element_ids is None else self.x_nodes[element_ids]

        # Reference points are constants: only the dependence on xi is traced
        xi = xi.detach().requires_grad_(True)

        # (N_e, N_p, dim)
        x = torch.einsum("en...,eqn...->eq...", x_nodes, self.sf.N(xi))

        # (N_e, N_p, dim, dim)
        return jacobian(xi, x)

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
        self.x_nodes = self._mesh.coordinates.at_elements()
