import torch.nn as nn

from neurom.fields.field_base import FieldBase
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)


class FieldLayout(nn.Module):
    """Container managing fields

    This container provides ways to access and manage the fields and their interpolations.

    Attributes:
        _fields : a nn.ModuleDict that holds all the fields that are added to the layout
        _interp (dict[str, QuadratureInterpolationResult]) : Holds the interpolation results for each field.
    Note:
        One cannot add() two fields with the same name nor update() an interpolation result of an unregistered field.
    """

    def __init__(self):
        super().__init__()
        self._fields = nn.ModuleDict()
        self._interp: dict[str, QuadratureInterpolationResult] = {}

    def add(self, field: FieldBase) -> FieldBase:
        """Adds a field to the layout

        Creates an entry with field.name in self._fields.

        Args:
            field (FieldBase): The field to register.
        Returns:
            The field we just registered.
        Raises:
            ValueError if the field.name is already present in the self._fields dictionnary.
        """
        if field.name in self._fields:
            raise ValueError(f"Field '{field.name}' already registered.")
        self._fields[field.name] = field
        return field

    def update(self, field: FieldBase, result: QuadratureInterpolationResult) -> None:
        """Update a field interpolation

        Modifies the entry in self._interp with field.name with the new result.

        Args:
            field (FieldBase): The field to which we will update the interpolation result.
            result (QuadratureInterpolationResult): The interpolation result to associate to the field.
        Raises:
            KeyError if the field.name is not present in the self._fields dictionnary.
        """
        if field.name not in self._fields:
            raise KeyError(f"No field named '{field.name}' registered.")
        self._interp[field.name] = result

    def __getitem__(self, name: str) -> QuadratureInterpolationResult:
        """Get the result of interpolation of a field

        Returns the entry in the self._interp dictionnary for the given name

        Args:
            name (str): The name of the field to get.
        Raises:
            KeyError if the field.name is not present in the self._fields dictionnary, i.e. it is not registered.
            RuntimeError if the field.name is not present in the self._interp dictionnary, i.e. interpolation was not computed.
        """
        if name not in self._fields:
            raise KeyError(f"No field named '{name}' registered.")
        if name not in self._interp:
            raise RuntimeError(f"Field '{name}' registered but not yet interpolated.")
        return self._interp[name]
