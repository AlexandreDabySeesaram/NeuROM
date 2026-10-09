"""Per-DOF-kind functionals ``ℓ`` -- the single extension point for elements.

Each functional evaluates its linear functional ``ℓ`` on an arbitrary function. It is used
in two places, so there is one concept rather than two:

- the transformation applies it to the **reference basis** -> a row of ``L_e``
  (``(L_e)_{ia} = ℓ_i(N_a)``), assembled then inverted to ``M_e``;
- ``FunctionSpace.interpolate(f)`` applies it to a **user function** -> that DOF's value.

Adding an element = declaring its ``dof_layout`` + (if a new kind) one functional here.
"""

from abc import ABC, abstractmethod

import torch


class DofFunctional(ABC):
    """A DOF kind's functional ``ℓ``, evaluable on the basis or on a user function."""

    #: Whether ``l_rows`` depends on the geometry (the mapping). ``False`` lets the
    #: transformation precompute a constant ``M_e`` once.
    geometry_dependent = False

    @abstractmethod
    def l_rows(self, element, mapping, xi, element_ids, n_e):
        """Rows of ``L_e`` for this kind's DOFs: ``ℓ_i`` applied to every basis function.

        Args:
            element (FiniteElement): The element (for its reference basis).
            mapping: Geometry mapping (used by geometry-dependent kinds).
            xi (torch.Tensor): Reference nodes of this kind's DOFs, ``(n_rows, dim_ref)``.
            element_ids (torch.Tensor | None): Element subset, or ``None`` for all.
            n_e (int): Number of elements.

        Returns:
            torch.Tensor: ``(n_e, n_rows, n_ref)``.
        """
        ...

    @abstractmethod
    def on_function(self, f, x_phys):
        """Apply ``ℓ`` to a user function ``f`` at this kind's physical DOF points.

        Args:
            f (callable): ``x (..., gdim) -> value (..., *value_shape)``.
            x_phys (torch.Tensor): Physical DOF points, ``(n_rows, gdim)``.

        Returns:
            torch.Tensor: ``(n_rows, *value_shape)`` -- ``ℓ_i(f)``.
        """
        ...


class ValueFunctional(DofFunctional):
    """Nodal value ``ℓ_i(w) = w(x_i)`` -- geometry-independent (`L` row = ``N(ξ_i)``)."""

    geometry_dependent = False

    def l_rows(self, element, mapping, xi, element_ids, n_e):
        xi_e = xi.unsqueeze(0).expand(n_e, -1, -1)  # (n_e, n_rows, dim_ref)
        return element.N(xi_e)  # (n_e, n_rows, n_ref)

    def on_function(self, f, x_phys):
        return f(x_phys)


class GradientFunctional(DofFunctional):
    """Nodal first derivative ``ℓ_i(w) = w'(x_i)`` (1-D; Rule 4, ``L`` row = ``N'(ξ_i)/J``)."""

    geometry_dependent = True

    def l_rows(self, element, mapping, xi, element_ids, n_e):
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
        return dN / J.unsqueeze(-1)

    def on_function(self, f, x_phys):
        x = x_phys.clone().requires_grad_(True)
        y = f(x)
        (dy,) = torch.autograd.grad(y.sum(), x, create_graph=False)
        return dy[
            ..., 0
        ]  # drop the single spatial axis -> matches the value convention


#: DOF kind -> functional. Extend with flux / normal_derivative for RT0 / Morley.
FUNCTIONALS = {
    "value": ValueFunctional(),
    "d1": GradientFunctional(),
}
