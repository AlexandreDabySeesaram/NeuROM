"""DOF transformation for elements whose DOFs are all nodal values."""

import torch

from neurom.dof_transformations.dof_transformation import DofTransformation


class IdentityDofTransformation(DofTransformation):
    """Identity transformation, used when every DOF is a nodal value.

    Nodal values are invariant under the reference-to-physical map, so
    physical and reference DOFs coincide (Lagrange elements).
    """

    def to_reference(
        self, u_elem: torch.Tensor, element_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the element DOFs unchanged.

        Args:
            u_elem (torch.Tensor): Physical DOFs gathered per element, shape
                ``(N_e, N_dofs, *f_shape)``.
            element_ids (torch.Tensor, optional): Unused.

        Returns:
            torch.Tensor: ``u_elem`` itself.
        """
        return u_elem
