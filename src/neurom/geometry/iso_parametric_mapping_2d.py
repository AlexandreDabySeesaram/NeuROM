"""Isoparametric mapping between reference and physical coordinates for 2D elements."""

import torch
import torch.nn as nn

from neurom.shape_functions.shape_function import ShapeFunction
from neurom.meshes.mesh import Mesh


class IsoparametricMapping2D(nn.Module):
    """Isoparametric mapping between reference and physical space for 2-D elements.

    Implements the forward map :math:`x = \\sum_n N_n(\\xi)\\, x_n` and
    the analytic inverse for linear triangular elements.  The Jacobian and
    its inverse are precomputed at construction time and refreshed by
    :meth:`update`.

    Attributes:
        sf (ShapeFunction): Shape-function object used to evaluate basis
            functions.
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
        """Inverse Jacobian matrices for each element.

        Returns:
            torch.Tensor: Inverse Jacobian matrices, shape ``(N_e, 2, 2)``.
        """
        return self._J_inv

    @property
    def det_jacobian(self):
        """Determinant of the Jacobian for each element.

        For a linear triangle element this equals twice the signed area of
        the physical triangle: :math:`\\det J = (b-a) \\times (c-a)`.

        Returns:
            torch.Tensor: Per-element Jacobian determinants, shape
            ``(N_e, 1)``.
        """
        return self._det_J

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

        Applies :math:`\\xi = J^{-1}(x - a)` where :math:`a` is the first
        node of each element.

        Args:
            x (torch.Tensor): Physical coordinates of shape
                ``(N_e, N_q, dim)``.

        Returns:
            torch.Tensor: Reference coordinates of shape ``(N_e, N_q, dim)``.

        Note:
            This inverse mapping is exact only for linear shape functions on
            triangular elements.
        """

        # Recover x shape
        _, N_q, _ = x.shape

        # Shape 'a' into (N_e, N_q, dim)
        a = self.x_nodes[:, 0, :].unsqueeze(1).expand(-1, N_q, -1)

        xi = torch.einsum("eql,elk->eqk", x - a, self.J_inv)

        return xi

    def inverse_map_at(self, x, element_ids):
        """Map physical positions to reference coordinates using a subset of elements.

        Same analytic inverse as :meth:`inverse_map`, but restricts the
        Jacobian lookup to the elements identified by ``element_ids``.

        Args:
            x (torch.Tensor): Physical coordinates of shape
                ``(N_e, N_q, dim)``.
            element_ids (torch.Tensor): Indices of the elements to use,
                shape ``(N_e,)``.

        Returns:
            torch.Tensor: Reference coordinates of shape ``(N_e, N_q, dim)``.

        Note:
            This inverse mapping is exact only for linear shape functions on
            triangular elements.
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
        """Refresh cached node positions and Jacobian from the underlying mesh.

        Must be called whenever the mesh node positions change (e.g. after a
        training step that moves the nodes) so that subsequent calls to
        :meth:`map`, :meth:`inverse_map`, :attr:`J_inv`, and
        :attr:`det_jacobian` use the updated geometry.
        """
        self.x_nodes = self._mesh.nodes_positions.at_elements()
        self._compute_J_inv()
