r"""Affine reference-to-physical mapping for P1-geometry simplices (bar, triangle, ...).

The geometry comes from the mesh's P1 coordinate element, so the map :math:`x(\xi)=\sum_n N_n(\xi)x_n`
is **affine**: its Jacobian :math:`J_e=\partial x/\partial\xi` is **constant per element**. This class
computes everything in closed form -- no autograd:

.. math::

    J_e[i,j] = \sum_n x_n[i]\,\frac{\partial N_n}{\partial\xi_j}, \qquad
    \det J_e,\ J_e^{-1}\ \text{in closed form}, \qquad
    \xi = \xi_0 + J_e^{-1}(x - x_0),

where :math:`\partial N_n/\partial\xi` is the (constant) reference gradient of the affine geometry
basis and :math:`(\xi_0, x_0)` is the reference/physical image of local vertex 0. Gradients still flow
to the node positions (the operations are plain differentiable tensor ops on ``x_nodes``), so
r-adaptivity is preserved without differentiating the map.

It is dimension-generic (bar, triangle, and higher simplices with ``gdim == dim_ref``); a single class
replaces the former per-dimension ``IsoparametricMapping1D`` / ``IsoparametricMapping2D``. A genuinely
curved / higher-order geometry (a :math:`\xi`-dependent :math:`J`) would be a separate mapping with an
autograd or Newton inverse -- the name "isoparametric" is reserved for that case.
"""

import torch
import torch.nn as nn

from neurom.meshes.mesh import Mesh


def _det_and_inv(J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Closed-form determinant and inverse of a batch of small square Jacobians.

    Closed forms (not :func:`torch.linalg.det` / :func:`torch.linalg.inv`) because the affine map's
    :math:`\det J` and :math:`J^{-1}` sit in the autograd graph that r-adaptivity backpropagates to
    node positions every step; the arithmetic forms have far cheaper, lighter-graph backward passes.
    Falls back to ``torch.linalg`` for ``dim_ref > 2`` (not currently reached).

    Args:
        J (torch.Tensor): Batched Jacobian, shape ``(N_e, n, n)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(det, inv)`` of shapes ``(N_e, 1)`` and ``(N_e, n, n)``.
    """
    n = J.shape[-1]
    if n == 1:
        det = J[..., 0, 0]  # (N_e,)
        inv = (1.0 / det).view(*J.shape[:-2], 1, 1)
        return det.unsqueeze(-1), inv
    if n == 2:
        a, b = J[..., 0, 0], J[..., 0, 1]
        c, d = J[..., 1, 0], J[..., 1, 1]
        det = a * d - b * c  # (N_e,)
        inv = torch.stack([d, -b, -c, a], dim=-1).view(*J.shape[:-2], 2, 2)
        inv = inv / det.unsqueeze(-1).unsqueeze(-1)
        return det.unsqueeze(-1), inv
    return torch.linalg.det(J).unsqueeze(-1), torch.linalg.inv(J)


def _apply_J_inv(J_inv: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    r"""Contract :math:`\xi_i = \sum_j (J^{-1})_{ij}\,\mathrm{offset}_j` with elementwise ops.

    A broadcast multiply-and-sum rather than ``einsum``/``matmul``: ``inverse_map`` sits in the
    double-backward path of r-adaptivity (the field gradient ``du/dx`` is differentiated, then
    backpropagated to node positions), where elementwise ops have a much lighter second-order graph
    than a contraction. For a 1-D map this is a plain scalar multiply.

    Args:
        J_inv (torch.Tensor): Inverse Jacobian, shape ``(N_e, dim_ref, gdim)``.
        offset (torch.Tensor): ``x - x0``, shape ``(N_e, N_q, gdim)``.

    Returns:
        torch.Tensor: Reference offset :math:`J^{-1}(x - x_0)`, shape ``(N_e, N_q, dim_ref)``.
    """
    # (N_e, 1, dim_ref, gdim) * (N_e, N_q, 1, gdim) -> sum over gdim -> (N_e, N_q, dim_ref)
    return (J_inv.unsqueeze(1) * offset.unsqueeze(-2)).sum(dim=-1)


class AffineSimplexMapping(nn.Module):
    r"""Affine reference-to-physical map built from a mesh's P1 coordinate element.

    Args:
        mesh (Mesh): Mesh whose coordinate field defines the physical domain; its coordinate
            element supplies the (affine) geometry basis.

    Attributes:
        sf (ShapeFunction): The geometry (coordinate) element's reference basis.
        x_nodes (torch.Tensor): Physical node coordinates per element, shape ``(N_e, n_nodes, gdim)``.
    """

    def __init__(self, mesh: Mesh):
        super().__init__()
        self.sf = mesh.coordinates.space.element.reference_basis
        self._mesh = mesh

        # Reference-cell constants (independent of node positions): cache them once so the per-step
        # update() only recomputes the x_nodes-dependent Jacobian. The affine geometry basis has a
        # constant reference gradient, evaluated here at a single probe point.
        cell = self.sf.reference_cell
        dtype = mesh.coordinates.full_values().dtype
        probe = cell.simplex[:1].unsqueeze(0).to(dtype=dtype)  # (1, 1, dim_ref)
        self._grad0 = self.sf.grad_N(probe)[0, 0]  # (n_nodes, dim_ref), constant for P1
        self._xi0 = cell.simplex[0].to(dtype=dtype)  # (dim_ref,) reference image of local vertex 0

        self._build()

    def _build(self) -> None:
        """Recompute the per-element Jacobian / inverse / determinant from the current node positions.

        Called per :meth:`update`; uses the cached constant reference gradient ``self._grad0`` so only
        the ``x_nodes``-dependent work is redone. ``J`` is constant per element (affine).
        """
        self.x_nodes = self._mesh.coordinates.at_elements()  # (N_e, n_nodes, gdim)
        grad0 = self._grad0.to(dtype=self.x_nodes.dtype, device=self.x_nodes.device)

        # J[e, i, j] = sum_n x_n[e, i] dN_n/dxi_j  -> (N_e, gdim, dim_ref), constant per element.
        self._J = torch.einsum("eni,nj->eij", self.x_nodes, grad0)
        self._det_J, self._J_inv = _det_and_inv(self._J)  # (N_e, 1), (N_e, dim_ref, gdim)

    def map(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Map reference coordinates to physical positions, :math:`x = \sum_n N_n(\xi)\,x_n`.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Physical positions, shape ``(N_e, N_q, gdim)``.
        """
        return torch.einsum("en...,eqn...->eq...", self.x_nodes, self.sf.N(xi))

    def jacobian_at(self, xi: torch.Tensor, element_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""Jacobian :math:`J_e=\partial x/\partial\xi`, broadcast over the query points.

        Constant per element (affine), so the cached :math:`J_e` is simply broadcast to the number of
        query points in ``xi``; the values do not depend on ``xi``.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, dim_ref)`` (only ``N_q`` is
                read, to size the broadcast).
            element_ids (torch.Tensor | None): Element subset ``(N_e,)``, or ``None`` for all.

        Returns:
            torch.Tensor: Jacobian, shape ``(N_e, N_q, gdim, dim_ref)``.
        """
        J = self._J if element_ids is None else self._J[element_ids]  # (N_e, gdim, dim_ref)
        return J.unsqueeze(1).expand(-1, xi.shape[1], -1, -1)  # (N_e, N_q, gdim, dim_ref)

    @property
    def det_jacobian(self) -> torch.Tensor:
        r"""Per-element Jacobian determinant :math:`\det J_e`, shape ``(N_e, 1)``.

        For a bar this is :math:`(x_1-x_0)/2`; for a triangle :math:`(b-a)\times(c-a)` (twice the
        signed area). Used by the quadrature measure as :math:`|\det J_e|\,w_q`.
        """
        return self._det_J

    @property
    def J_inv(self) -> torch.Tensor:
        r"""Per-element inverse Jacobian :math:`J_e^{-1}`, shape ``(N_e, dim_ref, gdim)``."""
        return self._J_inv

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Map physical positions to reference coordinates, :math:`\xi=\xi_0+J_e^{-1}(x-x_0)`.

        Exact for the affine map. ``x_0`` is local vertex 0's physical position, ``xi_0`` its
        reference position.

        Args:
            x (torch.Tensor): Physical coordinates, shape ``(N_e, N_q, gdim)``.

        Returns:
            torch.Tensor: Reference coordinates, shape ``(N_e, N_q, dim_ref)``.
        """
        x0 = self.x_nodes[:, 0, :].unsqueeze(1)  # (N_e, 1, gdim)
        xi0 = self._xi0.to(dtype=x.dtype, device=x.device)
        return xi0 + _apply_J_inv(self._J_inv, x - x0)

    def inverse_map_at(self, x: torch.Tensor, element_ids: torch.Tensor) -> torch.Tensor:
        r"""Inverse map restricted to ``element_ids`` (same closed form as :meth:`inverse_map`).

        Args:
            x (torch.Tensor): Physical coordinates, shape ``(n, N_q, gdim)`` (``n == len(element_ids)``).
            element_ids (torch.Tensor): Indices of the elements to use, shape ``(n,)``.

        Returns:
            torch.Tensor: Reference coordinates, shape ``(n, N_q, dim_ref)``.
        """
        x0 = self.x_nodes[element_ids, 0, :].unsqueeze(1)  # (n, 1, gdim)
        J_inv = self._J_inv[element_ids]  # (n, dim_ref, gdim)
        xi0 = self._xi0.to(dtype=x.dtype, device=x.device)
        return xi0 + _apply_J_inv(J_inv, x - x0)

    def update(self) -> None:
        """Refresh cached node positions and Jacobian from the underlying mesh.

        Must be called whenever the mesh node positions change (e.g. after a training step that moves
        the nodes) so subsequent calls use the updated geometry.
        """
        self._build()
