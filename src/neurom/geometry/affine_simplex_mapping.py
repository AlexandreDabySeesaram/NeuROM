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

from dataclasses import replace

from neurom.apply import apply
from neurom.meshes.mesh import Mesh
from neurom.samplings import ElementSampling, Sampling


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
        the ``x_nodes``-dependent work is redone. ``J`` is constant per element (affine). The per-element
        quantities are stored as :class:`~neurom.samplings.ElementSampling` (batch ``(N_e,)``) so the
        point-wise methods combine them with a points :class:`~neurom.samplings.Sampling` of *any*
        batch shape via :func:`~neurom.apply.apply`.
        """
        self.x_nodes = self._mesh.coordinates.at_elements()  # (N_e, n_nodes, gdim)
        grad0 = self._grad0.to(dtype=self.x_nodes.dtype, device=self.x_nodes.device)

        # J[e, i, j] = sum_n x_n[e, i] dN_n/dxi_j  -> (N_e, gdim, dim_ref), constant per element.
        J = torch.einsum("eni,nj->eij", self.x_nodes, grad0)
        det, J_inv = _det_and_inv(J)  # (N_e, 1), (N_e, dim_ref, gdim)
        self._J = ElementSampling(J)
        self._J_inv = ElementSampling(J_inv)
        self._det = ElementSampling(det)
        self._x0 = ElementSampling(self.x_nodes[:, 0, :])  # (N_e, gdim), local vertex 0

    def map(self, points: Sampling) -> Sampling:
        r"""Map reference coordinates to physical positions, :math:`x = \sum_n N_n(\xi)\,x_n`.

        Point-wise over whatever batch ``points`` carries (quadrature, arbitrary points, ...); the
        element axis is shared, every trailing axis is agnostic.

        Args:
            points (Sampling): Reference coordinates, ``f_shape == (dim_ref,)`` and a leading element
                batch axis.

        Returns:
            Sampling: Physical positions, same subtype/batch as ``points``, ``f_shape == (gdim,)``.
        """
        # Contract the nodes (n); the element axis (e) is shared and every further axis ("...") just
        # rides along -- batch-rank-agnostic. N comes from the basis, so this isn't an `apply` op.
        N = self.sf.N(points.values)  # (N_e, *pts, n_nodes)
        return replace(points, values=torch.einsum("eng,e...n->e...g", self.x_nodes, N))

    def jacobian_at(self, points: Sampling, element_ids: torch.Tensor | None = None) -> Sampling:
        r"""Jacobian :math:`J_e=\partial x/\partial\xi` at each of ``points`` (constant per element).

        Args:
            points (Sampling): Query points; only their batch shape matters (affine :math:`J` does not
                depend on the point coordinates).
            element_ids (torch.Tensor | None): Element subset ``(N_e,)``, or ``None`` for all.

        Returns:
            Sampling: Jacobian at each point, same subtype/batch as ``points``,
            ``f_shape == (gdim, dim_ref)``.
        """
        J = self._J.values if element_ids is None else self._J.values[element_ids]  # (N_e, gdim, dim_ref)
        # J is constant per element: broadcast it over the points' batch (singletons after the element
        # axis, then expand to the full batch so consumers see one J per query point).
        for _ in range(len(points.batch_shape) - 1):
            J = J.unsqueeze(1)
        J = J.expand(*points.batch_shape, *J.shape[-2:])
        return replace(points, values=J)

    @property
    def det_jacobian(self) -> Sampling:
        r"""Per-element Jacobian determinant :math:`\det J_e` as an ``ElementSampling`` (``f_shape (1,)``).

        For a bar this is :math:`(x_1-x_0)/2`; for a triangle :math:`(b-a)\times(c-a)` (twice the
        signed area). Used by the quadrature measure as :math:`|\det J_e|\,w_q`.
        """
        return self._det

    def inverse_map(self, x: Sampling) -> Sampling:
        r"""Map physical positions to reference coordinates, :math:`\xi=\xi_0+J_e^{-1}(x-x_0)`.

        Point-wise and batch-agnostic: the element-level :math:`J_e^{-1}` and :math:`x_0` are
        broadcast over every trailing axis of ``x``.

        Args:
            x (Sampling): Physical coordinates, ``f_shape == (gdim,)`` and a leading element axis.

        Returns:
            Sampling: Reference coordinates, same subtype/batch as ``x``, ``f_shape == (dim_ref,)``.
        """
        xi0 = self._xi0.to(dtype=x.values.dtype, device=x.values.device)
        return apply(
            lambda Ji, xq, x0: xi0 + torch.einsum("...ij,...j->...i", Ji, xq - x0),
            self._J_inv,
            x,
            self._x0,
        )

    def inverse_map_at(self, x: Sampling, element_ids: torch.Tensor) -> Sampling:
        r"""Inverse map restricted to ``element_ids`` (same point-wise form as :meth:`inverse_map`).

        Args:
            x (Sampling): Physical coordinates, leading axis of length ``len(element_ids)``.
            element_ids (torch.Tensor): Indices of the elements to use.

        Returns:
            Sampling: Reference coordinates, same subtype/batch as ``x``.
        """
        xi0 = self._xi0.to(dtype=x.values.dtype, device=x.values.device)
        J_inv = ElementSampling(self._J_inv.values[element_ids])
        x0 = ElementSampling(self._x0.values[element_ids])
        return apply(
            lambda Ji, xq, x0: xi0 + torch.einsum("...ij,...j->...i", Ji, xq - x0),
            J_inv,
            x,
            x0,
        )

    def update(self) -> None:
        """Refresh cached node positions and Jacobian from the underlying mesh.

        Must be called whenever the mesh node positions change (e.g. after a training step that moves
        the nodes) so subsequent calls use the updated geometry.
        """
        self._build()
