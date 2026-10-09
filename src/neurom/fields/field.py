"""Non-trainable field: fixed DOF values on a :class:`FunctionSpace`."""

from neurom.fields.field_base import FieldBase


class Field(FieldBase):
    """A field with fixed (non-trainable) DOF values.

    The values tensor has shape ``(n_scalar_dofs, n_components)`` and is stored as a
    buffer, so an optimizer never updates it. Use :class:`~neurom.fields.TrainableField`
    for learnable values.

    Args:
        space (FunctionSpace): The function space the field lives on.
        values (torch.Tensor): DOF values of shape ``(n_scalar_dofs, n_components)``.
        name (str): Human-readable identifier.

    Raises:
        ValueError: If ``values`` is not 2-D, or its row count differs from the space's
            scalar-DOF count.
    """

    def __init__(self, space, values, name: str = ""):
        super().__init__(space=space, name=name)
        if values.ndim != 2:
            raise ValueError(
                f"Field values must be (n_scalar_dofs, n_components); got {tuple(values.shape)}."
            )
        n_dofs = space.n_scalar_dofs
        if values.shape[0] != n_dofs:
            raise ValueError(
                f"Field values have {values.shape[0]} rows but the space has "
                f"{n_dofs} scalar DOFs."
            )
        self.register_buffer("values", values)

    def full_values(self):
        """Return the stored DOF values (no expansion needed)."""
        return self.values
