import torch.nn as nn

from neurom.fields.field_base import FieldBase
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)


class FieldLayout(nn.Module):
    """Container managing fields

    This container provides ways to access and manage the fields and their interpolations.
    """

    def __init__(self):
        super().__init__()
        self._fields = nn.ModuleDict()
        self._interp: dict[str, QuadratureInterpolationResult] = {}

    def add(self, field: FieldBase) -> FieldBase:
        if field.name in self._fields:
            raise ValueError(f"Field '{field.name}' already registered.")
        self._fields[field.name] = field
        return field

    def update(self, field: FieldBase, result: QuadratureInterpolationResult) -> None:
        if field.name not in self._fields:
            raise KeyError(f"No field named '{field.name}' registered.")
        self._interp[field.name] = result

    def __getitem__(self, name: str) -> QuadratureInterpolationResult:
        if name not in self._fields:
            raise KeyError(f"No field named '{name}' registered.")
        if name not in self._interp:
            raise RuntimeError(f"Field '{name}' registered but not yet interpolated.")
        return self._interp[name]
