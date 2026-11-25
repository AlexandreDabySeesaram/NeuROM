"""Abstract base class for fields living on a :class:`~neurom.function_space.FunctionSpace`."""

from abc import ABC, abstractmethod

import torch.nn as nn


class FieldBase(nn.Module, ABC):
    """Abstract base for every field -- a function living on a ``FunctionSpace``.

    A field holds one value per scalar degree of freedom of its space (with a trailing
    component axis for vector fields) and knows how to gather those values per cell via
    the space's DOF map. Concrete subclasses supply :meth:`full_values` (how the stored,
    possibly reduced, parameters expand to the full DOF vector).

    Args:
        space (FunctionSpace): The function space the field is defined on.
        name (str): Human-readable identifier (used by :class:`FieldLayout` for lookup).

    Attributes:
        space (FunctionSpace): The function space this field belongs to.
        name (str): The field's identifier.
    """

    def __init__(self, space, name: str = ""):
        super().__init__()
        self.space = space
        self.name = name

    @property
    def connectivity(self):
        """DOF connectivity ``(cell -> global scalar DOFs)`` taken from the space."""
        return self.space.connectivity

    @abstractmethod
    def full_values(self):
        """Return the complete DOF values.

        Returns:
            torch.Tensor: Values of shape ``(n_scalar_dofs, n_components)``.
        """
        ...

    @property
    def dim(self) -> int:
        """Number of value components per DOF."""
        return self.full_values().shape[1]

    def at_elements(self):
        """Gather the DOF values per cell through the space's DOF map.

        Returns:
            torch.Tensor: Values of shape ``(n_cells, n_local_dofs, n_components)``.
        """
        return self.full_values()[self.space.dof_map.cell_dofs]
