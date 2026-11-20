"""Trainable field: free DOF values are learnable parameters, fixed ones imposed by BCs."""

import torch.nn as nn

from neurom.fields.field_base import FieldBase


class TrainableField(FieldBase):
    """A field whose free DOF values are trainable parameters.

    The space compiles the boundary conditions (``bcs``) into a per-component free/fixed
    split: the free slots become the ``values_reduced`` :class:`~torch.nn.Parameter`, the
    fixed ones hold their imposed values. :meth:`full_values` scatters both back into the
    complete ``(n_scalar_dofs, n_components)`` tensor.

    Args:
        space (FunctionSpace): The function space the field lives on.
        init_values (torch.Tensor): Initial DOF values, shape
            ``(n_scalar_dofs, n_components)``. Only the free slots seed the parameter.
        bcs (Sequence[DirichletBC]): Essential boundary conditions; empty means all free.
        name (str): Human-readable identifier.

    Attributes:
        free_mask (torch.Tensor): Boolean ``(n_scalar_dofs, n_components)`` buffer, ``True``
            where the slot is trainable.
        imposed (torch.Tensor): ``(n_scalar_dofs, n_components)`` buffer of fixed values.
        values_reduced (torch.nn.Parameter): The free values. Shaped ``(n_free_dofs,
            n_components)`` when the free/fixed split is per-DOF (the common case: whole DOFs
            free or clamped together), or a flat ``(n_free_slots,)`` for a per-component
            roller where a DOF is partly free.
    """

    def __init__(self, space, init_values, bcs=(), name: str = ""):
        super().__init__(space=space, name=name)
        free_mask, imposed = space.compile_constraint(bcs, init_values.shape[1])
        self.register_buffer("imposed", imposed)

        # Whole-DOF split (every component of a DOF free or fixed together) keeps the
        # natural (n_free_dofs, n_components) shape; a per-component roller must flatten.
        free_dof = free_mask.all(dim=1)
        whole_dof = bool((free_dof | (~free_mask).all(dim=1)).all())
        self._whole_dof = whole_dof
        if whole_dof:
            self.register_buffer("free_mask", free_dof)  # (n_scalar_dofs,)
            self.values_reduced = nn.Parameter(init_values[free_dof])
        else:
            self.register_buffer(
                "free_mask", free_mask
            )  # (n_scalar_dofs, n_components)
            self.values_reduced = nn.Parameter(init_values[free_mask])

    def full_values(self):
        """Scatter the free parameter and the imposed values into the full DOF tensor."""
        full = self.imposed.clone()
        full[self.free_mask] = self.values_reduced.to(full.dtype)
        return full

    def freeze(self):
        """Stop training this field's free DOFs (``values_reduced`` requires no grad)."""
        self.values_reduced.requires_grad_(False)
