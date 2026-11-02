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
    """

    #: Whether :meth:`l_rows` depends on the geometry (the mapping). ``False`` lets the
    #: transformation precompute a constant :math:`M_e` once.
    geometry_dependent: bool = False

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
        return basis.N(xi)  # (n_rows, n_ref) -- N is batch-rank-agnostic, no fake batch axis needed


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
        self.direction = None if direction is None else tuple(float(d) for d in direction)

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
        grad_x = torch.einsum("eqji,eqaj->eqai", J_inv, ref_grad)  # (n_e,n_rows,n_ref,gdim)
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


#: DOF kind -> functional. Extend with flux / normal_derivative for RT0 / Morley.
FUNCTIONALS: dict[str, DofFunctional] = {
    "value": ValueFunctional(),
    "d1": GradientFunctional(),
}
