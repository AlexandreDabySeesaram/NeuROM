"""DOF transformation for 1-D elements carrying nodal first derivatives."""

import torch

from neurom.dof_transformations.dof_transformation import DofTransformation


class NodalDerivativeDofTransformation(DofTransformation):
    """Scale nodal first-derivative DOFs by the Jacobian of the mapping.

    For a 1-D element whose DOFs are nodal values and nodal first
    derivatives (e.g. Hermite beam), the chain rule gives
    :math:`\\frac{dw}{d\\xi} = J_e(\\xi_n)\\, \\frac{dw}{dx}`, hence
    :math:`M_e = \\mathrm{diag}(1, J_e(-1), 1, J_e(1))` for the Hermite beam.

    Attributes:
        mapping: Mapping providing ``jacobian_at(xi, element_ids)``.
        is_d1 (torch.Tensor): Boolean mask of the derivative DOFs, shape
            ``(N_dofs,)``.
        dof_nodes_xi (torch.Tensor): Reference coordinate of the node of
            each DOF, shape ``(N_dofs,)``.
    """

    _supported_kinds = ("value", "d1")

    def __init__(self, mapping, dof_kinds, dof_nodes_xi):
        """Initialise the transformation.

        Args:
            mapping: 1-D mapping providing ``jacobian_at(xi, element_ids)``.
            dof_kinds (Sequence[str]): Kind of each element DOF, ``"value"``
                or ``"d1"``.
            dof_nodes_xi (Sequence[float]): Reference coordinate of the node
                carrying each element DOF.

        Raises:
            ValueError: If a DOF kind is not supported, or if ``dof_kinds``
                and ``dof_nodes_xi`` have different lengths.
        """
        super().__init__()

        unknown = set(dof_kinds) - set(self._supported_kinds)
        if unknown:
            raise ValueError(
                f"Unsupported DOF kinds {sorted(unknown)}, expected {self._supported_kinds}."
            )
        if len(dof_kinds) != len(dof_nodes_xi):
            raise ValueError(
                f"Got {len(dof_kinds)} DOF kinds but {len(dof_nodes_xi)} DOF node coordinates."
            )

        self.mapping = mapping
        self.register_buffer(
            "is_d1", torch.tensor([kind == "d1" for kind in dof_kinds])
        )
        self.register_buffer("dof_nodes_xi", torch.tensor(dof_nodes_xi))

    def to_reference(
        self, u_elem: torch.Tensor, element_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scale the derivative DOFs of each element by its Jacobian.

        Args:
            u_elem (torch.Tensor): Physical DOFs gathered per element, shape
                ``(N_e, N_dofs, *f_shape)``.
            element_ids (torch.Tensor, optional): Indices of the elements
                ``u_elem`` belongs to, shape ``(N_e,)``.  All elements when
                ``None``.

        Returns:
            torch.Tensor: Reference DOFs, same shape as ``u_elem``.
        """
        n_e, n_dofs = u_elem.shape[:2]

        # (N_e, N_dofs, 1)
        xi = (
            self.dof_nodes_xi.to(u_elem.dtype).reshape(1, n_dofs, 1).expand(n_e, -1, -1)
        )

        # (N_e, N_dofs)
        J = self.mapping.jacobian_at(xi, element_ids)[..., 0, 0]

        # (N_e, N_dofs): J on derivative DOFs, 1 on value DOFs
        scale = torch.where(self.is_d1, J, torch.ones_like(J))

        # Broadcast over the field components
        return u_elem * scale.reshape(n_e, n_dofs, *([1] * (u_elem.ndim - 2)))
