r"""Per-DOF-kind functionals :math:`\ell` -- the single extension point for elements.

Each functional evaluates its linear functional :math:`\ell` on an arbitrary function. There is
one concept rather than three, because the same functional is used in three places:

- the transformation applies it to the **reference basis** -> a row of :math:`L_e`
  (:math:`(L_e)_{ia} = \ell_i(N_a)`), assembled then inverted to :math:`M_e`;
- ``interpolate`` / Dirichlet values apply it to a **user function** -> that DOF's value;
- the element's construction-time duality check applies the *reference* functional to the coded
  basis -> a row of :math:`V_{ab} = \hat\ell_a(\hat\Phi_b)`.

Adding an element = declaring its ``dof_layout`` + (if a new kind) one functional here.

Shape glossary (used throughout): ``n_e`` elements, ``n_rows`` the DOFs of one kind on the
element, ``n_ref`` reference basis functions, ``dim_ref`` the reference/parametric dimension
(1 bar, 2 triangle), ``gdim`` the geometric (ambient) dimension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import torch

from neurom.meshes.topology import EntityDim
from neurom.samplings import QuadratureSampling

if TYPE_CHECKING:
    from neurom.shape_functions.shape_function import ShapeFunction


class DofFunctional(ABC):
    r"""A DOF's functional :math:`\ell`, evaluable on the basis or on a user function.

    Each subclass implements one DOF *kind* (point value, first derivative, ...) by overriding the
    three methods below with the specific operator it applies.

    A functional operates on a *reference basis* (a :class:`~neurom.shape_functions.shape_function.ShapeFunction`)
    and a geometry *mapping* -- never on a :class:`~neurom.elements.finite_element.FiniteElement`.
    It is pure infrastructure: a library of operators that elements *reference* (by DOF kind), with
    no knowledge of the elements that use it, so the dependency runs one way (element -> functional
    -> basis).

    Two class-level booleans let the consumers pick the cheapest / correct path per functional
    (overridden by subclasses; see the inline comments on each for the full story):

    Attributes:
        geometry_dependent (bool): Whether :meth:`l_rows` depends on the geometry (the mapping).
            ``False`` (default) lets :class:`~neurom.dof_transformations.transformation.Transformation`
            precompute a constant :math:`M_e` once; ``True`` for e.g. the directional derivative.
        acts_pointwise (bool): Whether the functional is point-based -- fully determined by ``f`` at a
            single node, so ``interpolate`` drives it through :meth:`on_function`. ``True`` (default)
            for value / derivative DOFs; an *integral* functional (flux/moment) sets it ``False`` and
            is interpolated through its own entity-aware path instead.
    """

    #: Whether :meth:`l_rows` depends on the geometry (the mapping). ``False`` lets the
    #: transformation precompute a constant :math:`M_e` once.
    geometry_dependent: bool = False

    #: Whether this functional is **point-based** -- fully determined by ``f`` (and its jet) at a
    #: single physical node, so ``interpolate`` can drive it through :meth:`on_function`. ``True``
    #: for value / derivative DOFs. An *integral* functional (a flux/moment over an entity) sets this
    #: ``False``: its :meth:`on_function` is not meaningful (it needs the facet's normal/measure, not
    #: a point), and ``interpolate`` routes it through an entity-aware path instead. This is the one
    #: knob that keeps ``on_function``'s signature an honest point contract rather than carrying
    #: facet geometry every functional would have to ignore.
    acts_pointwise: bool = True

    @abstractmethod
    def l_rows(
        self,
        basis: ShapeFunction,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Rows of :math:`L_e` for this kind's DOFs: :math:`\ell_i` applied to every basis function.

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            mapping: Geometry mapping (``AffineSimplexMapping``), used by
                geometry-dependent kinds via ``mapping.jacobian_at``.
            xi (torch.Tensor): Reference nodes of this kind's DOFs, shape ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Element subset of shape ``(n_e,)``, or ``None`` for
                all elements.
            n_e (int): Number of elements.

        Returns:
            torch.Tensor: The rows of :math:`L_e` for this kind, shape ``(n_e, n_rows, n_ref)``.
        """
        ...

    @abstractmethod
    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply :math:`\ell` to a user function ``f`` at this kind's physical DOF points.

        The component axis is kept **explicit**: the result is always ``(n_rows, n_components)``
        (a scalar field is ``(n_rows, 1)``), matching the rest of the value pipeline -- no squeezed
        ``(n_rows,)``.

        Args:
            f (Callable): ``x (n_rows, gdim) -> value (n_rows, n_components)`` (a scalar field may
                also return ``(n_rows,)``; it is normalised to an explicit component axis).
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: :math:`\ell_i(f)`, shape ``(n_rows, n_components)``.
        """
        ...

    @abstractmethod
    def reference_row(self, basis: ShapeFunction, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference (geometry-free) functional applied to the reference basis at ``xi``.

        Builds rows of the duality matrix :math:`V_{ab} = \hat\ell_a(\hat\Phi_b)` used by the
        element's construction-time unisolvence/nodality check -- unlike :meth:`l_rows`, it carries
        no mapping (no :math:`J`), evaluating the functional on the reference cell with local
        orientation.

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            xi (torch.Tensor): Reference points of these DOFs, shape ``(n_rows, dim_ref)``.

        Returns:
            torch.Tensor: The duality rows for this kind, shape ``(n_rows, n_ref)``.
        """
        ...


class ValueFunctional(DofFunctional):
    r"""Nodal value :math:`\ell_i(w) = w(\mathbf x_i)` -- geometry-independent."""

    geometry_dependent: bool = False

    def l_rows(
        self,
        basis: ShapeFunction,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Point-value rows :math:`(L_e)_{ia} = N_a(\xi_i)`.

        The reference basis evaluated at this kind's DOF nodes, broadcast unchanged to every
        element -- a point value does not depend on the geometry, so ``mapping``/``element_ids``
        are unused.

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            mapping: Unused (point value is geometry-independent).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Unused.
            n_e (int): Number of elements to broadcast to.

        Returns:
            torch.Tensor: ``N(\xi)`` broadcast over elements, shape ``(n_e, n_rows, n_ref)``.
        """
        xi_e = xi.unsqueeze(0).expand(n_e, -1, -1)  # (n_e, n_rows, dim_ref)
        return basis.N(xi_e)  # (n_e, n_rows, n_ref)

    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluate the user function at the physical DOF points: :math:`\ell_i(f) = f(\mathbf x_i)`.

        The trailing axis is kept explicit: a scalar ``f`` returning ``(n_rows,)`` is normalised to
        ``(n_rows, 1)`` (flatten any ``value_shape`` to ``n_components``).

        Args:
            f (Callable): ``x (n_rows, gdim) -> value (n_rows, n_components)`` (``(n_rows,)`` scalar
                also accepted).
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: ``f`` at the points, shape ``(n_rows, n_components)``.
        """
        return f(x_phys).reshape(x_phys.shape[0], -1)  # (n_rows, n_components)

    def reference_row(self, basis: ShapeFunction, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference duality row :math:`N_a(\xi_i)` -- as :meth:`l_rows` but for a single element,
        geometry-free.

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)``.

        Returns:
            torch.Tensor: ``N(\xi)`` on the reference cell, shape ``(n_rows, n_ref)``.
        """
        return basis.N(
            xi
        )  # (n_rows, n_ref) -- N is batch-rank-agnostic, no fake batch axis needed


class GradientFunctional(DofFunctional):
    r"""Directional first derivative :math:`\ell_i(w) = \mathbf d\cdot\nabla w(\mathbf x_i)`.

    One functional family (plan section 3.3): the derivative of :math:`w` along a fixed physical
    direction :math:`\mathbf d`. Geometry-dependent (the physical gradient needs the mapping).

    Args:
        direction (Sequence[float] | None): The physical direction :math:`\mathbf d`, length
            ``gdim``. ``None`` (the default) selects the plain 1-D derivative
            :math:`\ell_i(w) = w'(\mathbf x_i)` -- the Hermite-beam ``d1`` case, where the single
            axis makes a direction redundant. A vector (e.g. ``[1, 0]`` for :math:`\mathbf e_x`)
            selects :math:`\mathbf d\cdot\nabla` in any dimension, via :math:`J_e^{-\top}`.
    """

    geometry_dependent: bool = True

    def __init__(self, direction=None) -> None:
        self.direction = (
            None if direction is None else tuple(float(d) for d in direction)
        )

    def l_rows(
        self,
        basis: ShapeFunction,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Directional-derivative rows :math:`(L_e)_{ia} = \mathbf d\cdot\nabla N_a(\xi_i)`.

        The reference gradient :math:`\partial N_a/\partial\xi` (autograd) pushed to the physical
        gradient with :math:`\nabla_x N = J_e^{-\top}\,\nabla_\xi N` and contracted with
        :math:`\mathbf d`. For the ``None`` (1-D plain-derivative) case this reduces to
        :math:`N_a'(\xi_i)/J_e`.

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            mapping: Geometry mapping; ``mapping.jacobian_at(xi, element_ids)`` supplies ``J_e``.
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Element subset ``(n_e,)``, or ``None`` for all.
            n_e (int): Number of elements.

        Returns:
            torch.Tensor: The rows of :math:`L_e` for this kind, shape ``(n_e, n_rows, n_ref)``.

        Raises:
            NotImplementedError: If ``direction is None`` and ``dim_ref != 1``, or if a directional
                derivative is requested on a non-square Jacobian (``gdim != dim_ref``).
        """
        xi_e = xi.unsqueeze(0).expand(n_e, -1, -1)  # (n_e, n_rows, dim_ref)
        ref_grad = basis.grad_N(xi_e)  # (n_e, n_rows, n_ref, dim_ref)

        if self.direction is None:
            if basis.reference_cell.dim_ref != 1:
                raise NotImplementedError(
                    "GradientFunctional without a direction is 1-D only; pass a direction "
                    "vector for the d.grad functional in higher dimensions."
                )
            J = mapping.jacobian_at(QuadratureSampling(xi_e), element_ids).values[
                ..., 0, 0
            ]  # (n_e, n_rows)
            return ref_grad[..., 0] / J.unsqueeze(-1)  # (n_e, n_rows, n_ref)

        J = mapping.jacobian_at(
            QuadratureSampling(xi_e), element_ids
        ).values  # (n_e, n_rows, gdim, dim_ref)
        if J.shape[-1] != J.shape[-2]:
            raise NotImplementedError(
                "directional derivative needs a square Jacobian (gdim == dim_ref)."
            )
        J_inv = torch.linalg.inv(J)  # (n_e, n_rows, dim_ref, gdim)
        # grad_x N_i = sum_j (J^{-1})_{ji} dN/dxi_j  (= (J^{-T} grad_xi N)_i)
        grad_x = torch.einsum(
            "eqji,eqaj->eqai", J_inv, ref_grad
        )  # (n_e,n_rows,n_ref,gdim)
        d = torch.tensor(self.direction, dtype=J.dtype, device=J.device)  # (gdim,)
        return torch.einsum("eqai,i->eqa", grad_x, d)  # (n_e, n_rows, n_ref)

    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""Directional derivative of ``f`` at the physical DOF points, via autograd.

        :math:`\ell_i(f) = \mathbf d\cdot\nabla f(\mathbf x_i)` (or :math:`f'(\mathbf x_i)` when
        ``direction is None``). The spatial axis is kept **explicit**: the result is ``(n_rows, 1)``
        -- for the 1-D case ``gdim == 1`` so the gradient already is ``(n_rows, 1)``; for a
        direction it is the projection :math:`\nabla f\cdot\mathbf d`, summed to ``(n_rows, 1)``.

        Args:
            f (Callable): ``x (n_rows, gdim) -> value (n_rows, 1)`` (a scalar field).
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: :math:`\mathbf d\cdot\nabla f` at the points, shape ``(n_rows, 1)``.
        """
        x = x_phys.clone().requires_grad_(True)
        y = f(x)
        (dy,) = torch.autograd.grad(y.sum(), x, create_graph=False)  # (n_rows, gdim)
        if self.direction is None:
            return dy  # (n_rows, 1) -- 1-D derivative, component axis kept explicit
        d = torch.tensor(self.direction, dtype=dy.dtype, device=dy.device)  # (gdim,)
        return (dy * d).sum(dim=-1, keepdim=True)  # (n_rows, 1)

    def reference_row(self, basis: ShapeFunction, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference duality row :math:`N_a'(\xi_i) = \mathrm dN_a/\mathrm d\xi` (1-D, no :math:`J`).

        Args:
            basis (ShapeFunction): The reference basis (evaluates ``N``).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, 1)``.

        Returns:
            torch.Tensor: :math:`\mathrm dN/\mathrm d\xi` on the reference cell, shape
            ``(n_rows, n_ref)``.

        Raises:
            NotImplementedError: If a direction is set (the reference duality convention for 2-D
                directional DOFs lands with the elements that use them, WP9) or ``dim_ref != 1``.
        """
        if self.direction is not None:
            raise NotImplementedError(
                "reference duality for a directional derivative is not defined yet "
                "(the 2-D directional elements that need it land in WP9)."
            )
        if basis.reference_cell.dim_ref != 1:
            raise NotImplementedError(
                "reference GradientFunctional without a direction is 1-D only."
            )
        # grad_N is batch-rank-agnostic: no fake batch axis. (n_rows, n_ref, dim_ref=1) -> take it.
        return basis.grad_N(xi)[..., 0]  # (n_rows, n_ref)


class FluxFunctional(DofFunctional):
    r"""Edge/face normal flux :math:`\ell_i(\mathbf v)=\int_{E_i}\mathbf v\cdot\mathbf n_i\,\mathrm ds` (RT0).

    The DOF of a contravariant (H(div)) element: the flux of a vector field through a facet. Two
    facts make it unusual among the functionals:

    - **the rows of** :math:`L_e` **are geometry-free.** The contravariant Piola map preserves
      normal fluxes (:math:`(\tfrac1{\det J}J\hat{\mathbf v})\cdot(R J\hat{\mathbf t}) =
      \hat{\mathbf v}\cdot(R\hat{\mathbf t})`, WP4), so the *physical* flux of a pushed-forward
      reference basis function equals its *reference* flux. Hence
      :math:`(L_e)_{ia}=\hat\ell_i(\hat H_a)` -- independent of the cell -- and ``l_rows`` is just
      the reference row broadcast over the elements (``geometry_dependent = False``, so the
      transformation precomputes a constant :math:`M_e=L_e^{-1}`; for the flux-nodal RT0 basis
      :math:`L_e=I`). The per-cell facet-orientation flip lives separately in
      :math:`T_e=\operatorname{diag}(\sigma)` (see :class:`~neurom.dof_transformations.transformation.Transformation`);
    - the **reference** functional is taken with the local cyclic-edge normal
      :math:`\hat{\mathbf n}_i=R\,\hat{\mathbf t}_i` (unnormalised, so the edge length folds into the
      measure), via a one-point (midpoint) rule -- exact because an RT0 field's
      :math:`\mathbf v\cdot\mathbf n` is affine along a straight edge.

    **Scope -- this is the RT0 flux functional, not a general H(div) moment functional.** It models
    exactly *one* zeroth normal moment per facet (weight :math:`\equiv 1`), evaluated with a
    single-point midpoint rule. Higher-order H(div) (RT1 / BDM) does **not** reuse it as-is: its
    facet DOFs are *several* moments against a :math:`P_k(E)` weight basis (the integrand reaches
    degree :math:`2k`, so the midpoint rule is no longer exact -- a :math:`(k{+}1)`-point Gauss rule
    is needed), it adds *interior* cell moments :math:`\int_K\mathbf v\cdot\mathbf q\,\mathrm dx` (a
    different functional kind), and the geometry-free :math:`L_e` above is special to the zeroth
    normal moment (weighted / interior moments carry Jacobian factors). Generalising means a *moment
    functional* (weight polynomial + quadrature rule + edge/interior support), deferred until an
    actual RT1/BDM element forces the shape. A **Nedelec** (H(curl)) element does **not** use this
    functional either: its DOF is the *tangential* circulation :math:`\int_E\mathbf v\cdot\hat{\mathbf t}`
    under the *covariant* map -- a sibling functional contracting with :math:`\hat{\mathbf t}` rather
    than :math:`\hat{\mathbf n}=R\hat{\mathbf t}` (in 2-D the two differ only by that rotation).
    """

    geometry_dependent: bool = False
    acts_pointwise: bool = (
        False  # an edge integral, not a point value -> entity-aware interpolation
    )

    def flux_values(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        midpoints: torch.Tensor,
        normals: torch.Tensor,
        measures: torch.Tensor,
    ) -> torch.Tensor:
        r"""Interpolate a vector field onto the flux DOFs: :math:`\ell_i(\mathbf f)=|E_i|\,\mathbf f(\mathbf m_i)\cdot\hat{\mathbf n}_i`.

        The one-point (midpoint) quadrature of :math:`\int_{E_i}\mathbf f\cdot\mathbf n\,\mathrm ds`
        with the **canonical** (global, ``lo -> hi``) facet normal and measure -- the same convention
        the stored flux DOF uses, so both cells sharing a facet compute the identical value and the
        per-cell orientation is left to :math:`T_e`. The entity-aware counterpart of
        :meth:`on_function` (this functional is ``acts_pointwise = False``); the caller
        (:func:`neurom.interpolation.interpolate`) supplies the facet geometry it gathers from the
        mesh.

        Args:
            f (Callable): The vector field, ``x (n, gdim) -> value (n, gdim)``.
            midpoints (torch.Tensor): Physical facet midpoints, shape ``(n_e, n_facets, gdim)``.
            normals (torch.Tensor): Canonical unit facet normals, shape ``(n_e, n_facets, gdim)``.
            measures (torch.Tensor): Facet measures :math:`|E|`, shape ``(n_e, n_facets, 1)``.

        Returns:
            torch.Tensor: Flux DOF values, shape ``(n_e, n_facets, 1)`` (one scalar per facet DOF).
        """
        n_e, n_f, gdim = midpoints.shape
        fval = f(midpoints.reshape(-1, gdim)).reshape(
            n_e, n_f, gdim
        )  # f at each midpoint
        return (fval * normals).sum(dim=-1, keepdim=True) * measures  # |E| (f . n_hat)

    @staticmethod
    def _reference_edge_normals(cell, xi: torch.Tensor) -> torch.Tensor:
        r"""Local cyclic-edge normals :math:`\hat{\mathbf n}=R\hat{\mathbf t}` for the rows at ``xi``.

        Unnormalised (``|n| = `` edge length, so the measure is folded in), with ``R`` the -90 deg
        rotation (``n = (t1, -t0)``). Each DOF node ``xi_i`` (an edge midpoint) is matched to its
        reference edge by that midpoint, so the result is independent of the row order / any subset.

        Args:
            cell (ReferenceCell): The reference cell supplying ``local_entities(EDGE)`` / ``simplex``.
            xi (torch.Tensor): DOF reference nodes (edge midpoints), shape ``(n_rows, dim_ref)``.

        Returns:
            torch.Tensor: Per-row reference normal, shape ``(n_rows, dim_ref)``.
        """
        edges = cell.local_entities(EntityDim.EDGE)  # (n_edges, 2) cyclic (a, b)
        v = cell.simplex.to(xi.dtype)  # (n_verts, dim_ref)
        mids = v[edges].mean(dim=1)  # (n_edges, dim_ref) edge midpoints
        t = v[edges[:, 1]] - v[edges[:, 0]]  # (n_edges, dim_ref) tangent a -> b
        normals = torch.stack([t[:, 1], -t[:, 0]], dim=-1)  # R t = (t1, -t0), -90 deg
        dist = torch.cdist(xi, mids)  # (n_rows, n_edges)
        idx = dist.argmin(dim=1)  # (n_rows,) the edge each DOF node is the midpoint of
        if not torch.allclose(
            dist.gather(1, idx[:, None]), torch.zeros_like(idx, dtype=xi.dtype)
        ):
            raise ValueError(
                "a flux DOF node is not an edge midpoint of the reference cell."
            )
        return normals[idx]  # (n_rows, dim_ref)

    def reference_row(self, basis: ShapeFunction, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference flux duality row :math:`\hat\ell_i(\hat H_a)=\hat H_a(\xi_i)\cdot\hat{\mathbf n}_i`.

        The (vector) reference basis at each DOF midpoint, contracted with that edge's local normal
        :math:`\hat{\mathbf n}_i=R\hat{\mathbf t}_i` -- the one-point flux. For the RT0 basis this is
        the identity (flux-nodal, :math:`V=I`).

        Args:
            basis (ShapeFunction): The (vector-valued) reference basis; ``N(xi)`` is
                ``(n_rows, n_ref, gdim)``.
            xi (torch.Tensor): DOF reference nodes (edge midpoints), shape ``(n_rows, dim_ref)``.

        Returns:
            torch.Tensor: The flux duality rows, shape ``(n_rows, n_ref)``.
        """
        n = self._reference_edge_normals(basis.reference_cell, xi)  # (n_rows, gdim)
        N = basis.N(xi)  # (n_rows, n_ref, gdim)
        return torch.einsum("rag,rg->ra", N, n.to(N.dtype))  # (n_rows, n_ref)

    def l_rows(
        self,
        basis: ShapeFunction,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Geometry-free flux rows: the reference row broadcast to every element.

        By the contravariant Piola flux-preservation property the physical flux of a pushed-forward
        basis function equals its reference flux, so :math:`(L_e)_{ia}=\hat\ell_i(\hat H_a)` does not
        depend on the cell (``mapping``/``element_ids`` unused); the per-cell orientation lives in
        :math:`T_e`.

        Args:
            basis (ShapeFunction): The (vector-valued) reference basis.
            mapping: Unused (the flux rows are geometry-free).
            xi (torch.Tensor): DOF reference nodes (edge midpoints), shape ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Unused.
            n_e (int): Number of elements to broadcast to.

        Returns:
            torch.Tensor: ``(n_e, n_rows, n_ref)``.
        """
        row = self.reference_row(basis, xi)  # (n_rows, n_ref)
        return row.unsqueeze(0).expand(n_e, -1, -1)  # (n_e, n_rows, n_ref)

    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""Not available through the base signature: the flux needs the physical facet geometry.

        :math:`\ell_i(\mathbf f)=\int_{E_i}\mathbf f\cdot\mathbf n_i\,\mathrm ds\approx
        |E_i|\,\mathbf f(\mathbf m_i)\cdot\hat{\mathbf n}_i` requires the physical normal
        :math:`\hat{\mathbf n}_i` and measure :math:`|E_i|` of each facet, which the point-only
        ``(f, x_phys)`` signature cannot supply. Interpolating a field onto an RT0 space therefore
        goes through the entity-geometry-aware path (WP9 stage C wiring), not this method.
        """
        raise NotImplementedError(
            "FluxFunctional is not point-based (acts_pointwise=False): on_function needs the physical "
            "facet normal and measure, which a single point cannot carry. interpolate() routes it "
            "through the entity-aware FluxFunctional.flux_values instead."
        )


#: DOF kind -> functional. Extend with normal_derivative for Morley.
FUNCTIONALS: dict[str, DofFunctional] = {
    "value": ValueFunctional(),
    "d1": GradientFunctional(),
    "flux": FluxFunctional(),
}
