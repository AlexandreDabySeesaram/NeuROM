"""The one transformation: assemble L_e from per-kind functionals, invert to M_e, apply.

``u_phys = L_e q_ref`` with ``(L_e)_{ia} = ℓ_i(N_a)`` (each DOF's functional applied to the
reference basis). Then ``q_ref = M_e u_phys`` with ``M_e = L_e^{-1}`` (square, unisolvent) or
``pinv(L_e)`` (rectangular ``n_phys × n_ref``). The single central inversion is what lets a
*coupled* element (Morley) work: its functional fills coupled rows and the full L_e is inverted.

Short-circuits (no per-forward work):
- if ``L_e ≡ I`` (nodal-interpolatory Lagrange / DG0) -> identity, return the DOFs unchanged;
- if no kind is geometry-dependent -> ``M_e`` is constant, computed once.
Otherwise (e.g. the Hermite ``d1`` block depends on ``J``) ``L_e`` is assembled and inverted
each call, in-graph so position gradients flow (r-adaptivity).
"""

import torch
import torch.nn as nn

from neurom.dof_transformations.functionals import FUNCTIONALS


def _invert(L: torch.Tensor) -> torch.Tensor:
    """Invert ``L`` (``(e, p, r)``) to ``M`` (``(e, r, p)``): inverse if square, else pinv."""
    return torch.linalg.inv(L) if L.shape[-1] == L.shape[-2] else torch.linalg.pinv(L)


class Transformation(nn.Module):
    """Maps physical element DOFs to reference coefficients: ``q_ref = M_e u_phys``.

    Args:
        element (FiniteElement): Supplies ``dof_kinds`` / ``dof_nodes_xi`` / basis.
        mapping: Geometry mapping passed to geometry-dependent functionals.
    """

    def __init__(self, element, mapping):
        super().__init__()
        self.element = element
        self.mapping = mapping

        kinds = element.dof_kinds
        unknown = set(kinds) - set(FUNCTIONALS)
        if unknown:
            raise NotImplementedError(
                f"No functional for DOF kind(s) {sorted(unknown)}."
            )

        groups: dict[str, list] = {}
        for i, k in enumerate(kinds):
            groups.setdefault(k, []).append(i)
        self._groups = {k: torch.tensor(idx) for k, idx in groups.items()}
        self._xi = element.dof_nodes_xi()  # (n_phys, dim_ref)
        self._n_phys = element.n_dofs
        self._n_ref = element.n_ref

        self._geometry_dependent = any(
            FUNCTIONALS[k].geometry_dependent for k in groups
        )

        # Geometry-independent -> M_e is constant; precompute once (and detect identity).
        self._identity = False
        M_const = None
        if not self._geometry_dependent:
            L = self._assemble_L(1, self._xi.dtype, self._xi.device, None)
            M = _invert(L)[0]  # (n_ref, n_phys)
            if M.shape[0] == M.shape[1] and torch.allclose(
                M, torch.eye(M.shape[0], dtype=M.dtype)
            ):
                self._identity = True
            else:
                M_const = M
        self.register_buffer("_M_const", M_const)

    def _assemble_L(self, n_e, dtype, device, element_ids):
        L = torch.zeros(n_e, self._n_phys, self._n_ref, dtype=dtype, device=device)
        for kind, rows in self._groups.items():
            xi = self._xi[rows].to(dtype=dtype, device=device)
            L[:, rows, :] = FUNCTIONALS[kind].l_rows(
                self.element, self.mapping, xi, element_ids, n_e
            )
        return L

    def to_reference(self, u_elem: torch.Tensor, element_ids=None) -> torch.Tensor:
        """Apply ``M_e`` to the physical element DOFs; broadcasts over value components."""
        if self._identity:
            return u_elem
        if self._M_const is not None:
            return torch.einsum(
                "rp,ep...->er...", self._M_const.to(u_elem.dtype), u_elem
            )
        L = self._assemble_L(u_elem.shape[0], u_elem.dtype, u_elem.device, element_ids)
        return torch.einsum("erp,ep...->er...", _invert(L), u_elem)
