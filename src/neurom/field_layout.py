"""Container mapping field names to their interpolation results."""

import torch.nn as nn

from neurom.fields.field_base import FieldBase
from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult


class FieldLayout(nn.Module):
    """Container managing registered fields and their interpolation results.

    This container provides ways to register fields, update their quadrature
    interpolation results, and retrieve those results by field name.
    It is not allowed to register two fields with the same name, nor to update
    the interpolation result of a field that has not been registered first.

    Attributes:
        _fields (nn.ModuleDict): Holds all fields registered via :meth:`add`.
        _interp (dict[str, QuadratureAssemblyResult]): Maps field names to
            their latest interpolation results, populated by :meth:`update`.
    """

    def __init__(self):
        super().__init__()
        self._fields = nn.ModuleDict()
        self._interp: dict[str, QuadratureAssemblyResult] = {}

    def add(self, field: FieldBase) -> FieldBase:
        """Add a field to the layout.

        Creates an entry keyed by ``field.name`` in ``self._fields``.

        Args:
            field (FieldBase): The field to register.

        Returns:
            FieldBase: The field that was just registered.

        Raises:
            ValueError: If a field with the same name is already present in
                ``self._fields``.
        """
        if field.name in self._fields:
            raise ValueError(f"Field '{field.name}' already registered.")
        self._fields[field.name] = field
        return field

    def update(self, field: FieldBase, result: QuadratureAssemblyResult) -> None:
        """Update the interpolation result stored for a registered field.

        Replaces (or creates) the entry in ``self._interp`` keyed by
        ``field.name`` with the new ``result``.

        Args:
            field (FieldBase): The field whose interpolation result should be
                updated.
            result (QuadratureAssemblyResult): The new interpolation result to
                associate with the field.

        Raises:
            KeyError: If ``field.name`` is not present in ``self._fields``,
                i.e. the field has not been registered via :meth:`add`.
        """
        if field.name not in self._fields:
            raise KeyError(f"No field named '{field.name}' registered.")
        self._interp[field.name] = result

    def __getitem__(self, name: str) -> QuadratureAssemblyResult:
        """Return the interpolation result for a registered field.

        Looks up the entry in ``self._interp`` keyed by ``name``.

        Args:
            name (str): The name of the field whose interpolation result is
                requested.

        Returns:
            QuadratureAssemblyResult: The interpolation result previously
            stored for the field with the given name.

        Raises:
            KeyError: If ``name`` is not present in ``self._fields``, i.e. the
                field has not been registered via :meth:`add`.
            RuntimeError: If ``name`` is registered but not yet present in
                ``self._interp``, i.e. interpolation has not been computed yet.
        """
        if name not in self._fields:
            raise KeyError(f"No field named '{name}' registered.")
        if name not in self._interp:
            raise RuntimeError(f"Field '{name}' registered but not yet interpolated.")
        return self._interp[name]
