"""Abstract base class for DOF transformations."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DofTransformation(nn.Module, ABC):
    """Map physical element DOFs to the reference DOFs of the shape function.

    Global DOFs are stored in physical form (e.g. ``w`` and ``dw/dx``) so
    that they are shared consistently between elements.  The shape functions
    are defined on the reference element and expect reference DOFs (e.g.
    ``w`` and ``dw/dxi``).  A ``DofTransformation`` performs this per-element
    change of basis :math:`u^{ref}_e = M_e\\, u^{phys}_e`.
    """

    @abstractmethod
    def to_reference(
        self, u_elem: torch.Tensor, element_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform physical element DOFs into reference element DOFs.

        Args:
            u_elem (torch.Tensor): Physical DOFs gathered per element, shape
                ``(N_e, N_dofs, *f_shape)``.
            element_ids (torch.Tensor, optional): Indices of the elements
                ``u_elem`` belongs to, shape ``(N_e,)``.  All elements when
                ``None``.

        Returns:
            torch.Tensor: Reference DOFs, same shape as ``u_elem``.
        """
        pass
