r"""The one transformation: assemble :math:`L_e` from the per-DOF functionals, invert to :math:`M_e`.

The field on a cell is written from *reference coefficients* :math:`q_{\mathrm{ref}}`, but the
stored DOFs are *physical* values :math:`u_{\mathrm{phys}}` (e.g. a physical derivative for Hermite).
They are related by

.. math::

    u_{\mathrm{phys}} = L_e\, q_{\mathrm{ref}}, \qquad (L_e)_{ia} = \ell_i(N_a),

(each DOF's functional applied to the reference basis), so

.. math::

    q_{\mathrm{ref}} = M_e\, u_{\mathrm{phys}}, \qquad M_e = L_e^{-1}

(square/unisolvent) or :math:`\operatorname{pinv}(L_e)` (rectangular ``n_phys x n_ref``). The single
central inversion is what lets a *coupled* element (Morley) work: its functional fills coupled rows
and the full :math:`L_e` is inverted.

Short-circuits (no per-forward work):

- if :math:`L_e = I` (nodal-interpolatory Lagrange / DG0) -> identity, return the DOFs unchanged;
- if no kind is geometry-dependent -> :math:`M_e` is constant, computed once.

Otherwise (e.g. the Hermite ``d1`` block depends on :math:`J`) :math:`L_e` is assembled and inverted
each call, in-graph so position gradients flow (r-adaptivity).

Shape glossary: ``n_e`` elements, ``n_phys`` physical DOFs per element (``= element.n_dofs``),
``n_ref`` reference basis functions, ``dim_ref`` the reference/parametric dimension (1 bar,
2 triangle). ``value_shape`` is the field's component shape (``()`` scalar, ``(2,)`` 2-vector).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from neurom.dof_transformations.functionals import DofFunctional
    from neurom.elements.finite_element import FiniteElement


def _invert(L: torch.Tensor) -> torch.Tensor:
    r"""Invert :math:`L_e` to :math:`M_e`: exact inverse if square, else the pseudo-inverse.

    Args:
        L (torch.Tensor): Batched :math:`L_e`, shape ``(n_e, n_phys, n_ref)``.

    Returns:
        torch.Tensor: Batched :math:`M_e`, shape ``(n_e, n_ref, n_phys)`` -- ``torch.linalg.inv``
        when ``n_phys == n_ref`` (square, unisolvent), otherwise ``torch.linalg.pinv``.
    """
    return torch.linalg.inv(L) if L.shape[-1] == L.shape[-2] else torch.linalg.pinv(L)


class Transformation(nn.Module):
    r"""Maps physical element DOFs to reference coefficients: :math:`q_{\mathrm{ref}} = M_e u_{\mathrm{phys}}`.

    Args:
        element (FiniteElement): Supplies ``dof_kinds`` / ``dof_nodes_xi`` / the reference basis.
        mapping: Geometry mapping (``IsoparametricMapping1D``/``2D``) passed to geometry-dependent
            functionals (used via ``mapping.jacobian_at``).

    Attributes:
        element (FiniteElement): The element being transformed.
        mapping: The geometry mapping.
        _groups (dict[DofFunctional, torch.Tensor]): functional -> the local DOF indices sharing
            it, each an index tensor of shape ``(n_rows,)``.
        _xi (torch.Tensor): Reference node of each DOF, shape ``(n_phys, dim_ref)``.
        _n_phys (int): Physical DOFs per element.
        _n_ref (int): Reference basis functions.
        _geometry_dependent (bool): Whether any DOF kind's functional depends on the geometry.
        _identity (bool): Whether :math:`M_e = I` (short-circuit: return the DOFs unchanged).
        _M_const (torch.Tensor | None): The constant :math:`M_e`, shape ``(n_ref, n_phys)``, when
            geometry-independent and not identity; otherwise ``None`` (a registered buffer).
    """

    def __init__(self, element: FiniteElement, mapping) -> None:
        super().__init__()
        self.element = element
        self.mapping = mapping

        # Every DOF must carry a functional.
        layout = element.dof_layout
        missing = sorted({d.kind for d in layout if d.functional is None})
        if missing:
            raise NotImplementedError(f"No functional for DOF kind(s) {missing}.")

        # Group the local DOF indices by functional (DOFs sharing one are batched).
        groups: dict[DofFunctional, list[int]] = {}
        for i, d in enumerate(layout):
            groups.setdefault(d.functional, []).append(i)
        self._groups = {func: torch.tensor(idx) for func, idx in groups.items()}
        self._xi = element.dof_nodes_xi()  # (n_phys, dim_ref)
        self._n_phys = element.n_dofs
        self._n_ref = element.n_ref

        # Does any functional depend on the geometry (the mapping)?
        self._geometry_dependent = any(f.geometry_dependent for f in groups)

        # Geometry-independent -> M_e is constant; precompute it once (and detect identity).
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

    def _assemble_L(
        self,
        n_e: int,
        dtype: torch.dtype,
        device: torch.device,
        element_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""Assemble :math:`L_e` by filling, per kind, the rows :math:`(L_e)_{ia} = \ell_i(N_a)`.

        Args:
            n_e (int): Number of elements to assemble for.
            dtype (torch.dtype): dtype of the assembled matrix.
            device (torch.device): device of the assembled matrix.
            element_ids (torch.Tensor | None): Element subset ``(n_e,)``, or ``None`` for all
                (passed through to geometry-dependent functionals).

        Returns:
            torch.Tensor: :math:`L_e`, shape ``(n_e, n_phys, n_ref)``.
        """
        L = torch.zeros(n_e, self._n_phys, self._n_ref, dtype=dtype, device=device)
        for func, rows in self._groups.items():
            xi = self._xi[rows].to(dtype=dtype, device=device)  # (n_rows, dim_ref)
            L[:, rows, :] = func.l_rows(
                self.element.reference_basis, self.mapping, xi, element_ids, n_e
            )  # (n_e, n_rows, n_ref)
        return L

    def to_reference(
        self, u_elem: torch.Tensor, element_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Apply :math:`M_e` to the physical element DOFs; broadcasts over value components.

        Computes :math:`q_{\mathrm{ref}} = M_e\, u_{\mathrm{phys}}` per element, taking the cheapest
        available path: identity (return unchanged), the precomputed constant :math:`M_e`, or
        assemble-and-invert :math:`L_e` in-graph (geometry-dependent kinds).

        Args:
            u_elem (torch.Tensor): Physical element DOFs, shape ``(n_e, n_phys, *value_shape)``.
            element_ids (torch.Tensor | None): Element subset ``(n_e,)``, or ``None`` for all.

        Returns:
            torch.Tensor: Reference coefficients, shape ``(n_e, n_ref, *value_shape)``.
        """
        if self._identity:
            return u_elem
        if self._M_const is not None:
            return torch.einsum(
                "rp,ep...->er...", self._M_const.to(u_elem.dtype), u_elem
            )
        L = self._assemble_L(u_elem.shape[0], u_elem.dtype, u_elem.device, element_ids)
        return torch.einsum("erp,ep...->er...", _invert(L), u_elem)
