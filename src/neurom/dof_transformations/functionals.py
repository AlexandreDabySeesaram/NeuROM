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

if TYPE_CHECKING:
    from neurom.elements.finite_element import FiniteElement


class DofFunctional(ABC):
    r"""A DOF's functional :math:`\ell`, evaluable on the basis or on a user function.

    Each subclass implements one DOF *kind* (point value, first derivative, ...) by overriding the
    three methods below with the specific operator it applies.
    """

    #: Whether :meth:`l_rows` depends on the geometry (the mapping). ``False`` lets the
    #: transformation precompute a constant :math:`M_e` once.
    geometry_dependent: bool = False

    @abstractmethod
    def l_rows(
        self,
        element: FiniteElement,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Rows of :math:`L_e` for this kind's DOFs: :math:`\ell_i` applied to every basis function.

        Args:
            element (FiniteElement): The element (for its reference basis).
            mapping: Geometry mapping (``IsoparametricMapping1D``/``2D``), used by
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

        Args:
            f (Callable): ``x (..., gdim) -> value (..., *value_shape)``.
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: :math:`\ell_i(f)`, shape ``(n_rows, *value_shape)``.
        """
        ...

    @abstractmethod
    def reference_row(self, element: FiniteElement, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference (geometry-free) functional applied to the reference basis at ``xi``.

        Builds rows of the duality matrix :math:`V_{ab} = \hat\ell_a(\hat\Phi_b)` used by the
        element's construction-time unisolvence/nodality check -- unlike :meth:`l_rows`, it carries
        no mapping (no :math:`J`), evaluating the functional on the reference cell with local
        orientation.

        Args:
            element (FiniteElement): The element (for its reference basis).
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
        element: FiniteElement,
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
            element (FiniteElement): The element (for its reference basis ``N``).
            mapping: Unused (point value is geometry-independent).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Unused.
            n_e (int): Number of elements to broadcast to.

        Returns:
            torch.Tensor: ``N(\xi)`` broadcast over elements, shape ``(n_e, n_rows, n_ref)``.
        """
        xi_e = xi.unsqueeze(0).expand(n_e, -1, -1)  # (n_e, n_rows, dim_ref)
        return element.N(xi_e)  # (n_e, n_rows, n_ref)

    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluate the user function at the physical DOF points: :math:`\ell_i(f) = f(\mathbf x_i)`.

        Args:
            f (Callable): ``x (n_rows, gdim) -> value (n_rows, *value_shape)``.
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: ``f`` at the points, shape ``(n_rows, *value_shape)``.
        """
        return f(x_phys)

    def reference_row(self, element: FiniteElement, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference duality row :math:`N_a(\xi_i)` -- as :meth:`l_rows` but for a single element,
        geometry-free.

        Args:
            element (FiniteElement): The element (for its reference basis ``N``).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)``.

        Returns:
            torch.Tensor: ``N(\xi)`` on the reference cell, shape ``(n_rows, n_ref)``.
        """
        return element.N(xi.unsqueeze(0))[0]  # (n_rows, n_ref)


class GradientFunctional(DofFunctional):
    r"""Nodal first derivative :math:`\ell_i(w) = w'(\mathbf x_i)` -- 1-D, geometry-dependent."""

    geometry_dependent: bool = True

    def l_rows(
        self,
        element: FiniteElement,
        mapping,
        xi: torch.Tensor,
        element_ids: torch.Tensor | None,
        n_e: int,
    ) -> torch.Tensor:
        r"""Physical-derivative rows :math:`(L_e)_{ia} = N_a'(\xi_i)\,/\,J_e`.

        The reference basis derivative :math:`\mathrm dN_a/\mathrm d\xi` at the DOF nodes (via
        autograd) divided by the (1-D) cell Jacobian :math:`J_e` from ``mapping``. Raises for
        ``dim_ref > 1`` (2-D would need :math:`J_e^{-\top}`).

        Args:
            element (FiniteElement): The element (for its reference basis ``N``).
            mapping: Geometry mapping; ``mapping.jacobian_at(xi, element_ids)`` supplies ``J_e``.
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, dim_ref)`` with ``dim_ref == 1``.
            element_ids (torch.Tensor | None): Element subset ``(n_e,)``, or ``None`` for all.
            n_e (int): Number of elements.

        Returns:
            torch.Tensor: The rows of :math:`L_e` for this kind, shape ``(n_e, n_rows, n_ref)``.

        Raises:
            NotImplementedError: If ``element.dim_ref != 1``.
        """
        if element.dim_ref != 1:
            raise NotImplementedError(
                "GradientFunctional is 1-D only (2-D needs J^{-T})."
            )
        xi_e = xi.unsqueeze(0).expand(n_e, -1, -1).clone().requires_grad_(True)
        N = element.N(xi_e)  # (n_e, n_rows, n_ref)
        n_ref = N.shape[-1]
        dN = torch.stack(
            [
                torch.autograd.grad(
                    N[..., a].sum(), xi_e, create_graph=True, retain_graph=True
                )[0][..., 0]
                for a in range(n_ref)
            ],
            dim=-1,
        )  # (n_e, n_rows, n_ref)
        J = mapping.jacobian_at(xi_e, element_ids)[..., 0, 0]  # (n_e, n_rows)
        return dN / J.unsqueeze(-1)  # (n_e, n_rows, n_ref)

    def on_function(
        self, f: Callable[[torch.Tensor], torch.Tensor], x_phys: torch.Tensor
    ) -> torch.Tensor:
        r"""First derivative of the user function at the physical DOF points,
        :math:`\ell_i(f) = f'(\mathbf x_i)`, via autograd; drops the single spatial axis to match
        the value convention.

        Args:
            f (Callable): ``x (n_rows, 1) -> value (n_rows,)``.
            x_phys (torch.Tensor): Physical DOF points, shape ``(n_rows, 1)``.

        Returns:
            torch.Tensor: :math:`f'` at the points, shape ``(n_rows,)``.
        """
        x = x_phys.clone().requires_grad_(True)
        y = f(x)
        (dy,) = torch.autograd.grad(y.sum(), x, create_graph=False)
        return dy[..., 0]

    def reference_row(self, element: FiniteElement, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference duality row :math:`N_a'(\xi_i) = \mathrm dN_a/\mathrm d\xi` at the DOF nodes --
        the reference derivative with **no** :math:`J` (unlike :meth:`l_rows`). Raises for
        ``dim_ref > 1``.

        Args:
            element (FiniteElement): The element (for its reference basis ``N``).
            xi (torch.Tensor): DOF reference nodes, shape ``(n_rows, 1)``.

        Returns:
            torch.Tensor: :math:`\mathrm dN/\mathrm d\xi` on the reference cell, shape
            ``(n_rows, n_ref)``.

        Raises:
            NotImplementedError: If ``element.dim_ref != 1``.
        """
        if element.dim_ref != 1:
            raise NotImplementedError(
                "reference GradientFunctional is 1-D only (2-D: J^{-T})."
            )
        x = xi.unsqueeze(0).clone().requires_grad_(True)  # (1, n_rows, 1)
        N = element.N(x)  # (1, n_rows, n_ref)
        n_ref = N.shape[-1]
        dN = torch.stack(
            [
                torch.autograd.grad(
                    N[..., a].sum(), x, create_graph=False, retain_graph=True
                )[0][..., 0]
                for a in range(n_ref)
            ],
            dim=-1,
        )  # (1, n_rows, n_ref) -- reference derivative dN/dxi (no J)
        return dN[0]  # (n_rows, n_ref)


#: DOF kind -> functional. Extend with flux / normal_derivative for RT0 / Morley.
FUNCTIONALS: dict[str, DofFunctional] = {
    "value": ValueFunctional(),
    "d1": GradientFunctional(),
}
