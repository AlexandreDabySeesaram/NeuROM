import pytest
import torch

from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)

torch.set_default_dtype(torch.float32)


class DummyField(FieldBase):
    """Minimal concrete ``FieldBase`` implementation for testing purposes.

    The topology argument is not needed for the current tests, so ``None`` is
    passed to the superclass.
    """

    def __init__(self, name: str):
        # Provide a dummy topology; ``None`` is acceptable for these unit tests.
        super().__init__(name=name, topology=None)

    def full_values(self):
        return torch.tensor([])

    def at_elements(self):
        return torch.tensor([])


def test_add_and_duplicate():
    """Adding a field works and duplicate addition raises ``ValueError``.

    The first call to ``add`` should succeed, while a second call with the
    same field name must raise.
    """
    layout = FieldLayout()
    field = DummyField(name="temperature")

    # First addition should succeed.
    layout.add(field)

    # Adding a field with the same name should raise ``ValueError``.
    with pytest.raises(ValueError, match="already registered"):
        layout.add(field)


def test_add_returns_field():
    """Verify that ``add`` returns the exact field instance that was added."""
    layout = FieldLayout()
    field = DummyField(name="pressure")
    result = layout.add(field)

    assert result is field

    # Ensure the field is now registered inside the layout.
    assert field.name in layout._fields


def test_update_and_access():
    """After ``update`` the stored interpolation can be accessed via ``[]``.

    The test creates a dummy ``QuadratureInterpolationResult`` and checks
    that ``layout[field.name]`` returns the same object.
    """
    layout = FieldLayout()
    field = DummyField(name="displacement")
    layout.add(field)

    # Create a dummy interpolation result.
    result = QuadratureInterpolationResult(
        x=torch.randn(2, 3, 1),
        u=torch.randn(2, 3, 1),
        measure=torch.randn(2, 3, 1),
    )

    # ``update`` should store the result.
    layout.update(field, result)

    # Access via ``__getitem__`` should return the same object.
    retrieved = layout[field.name]
    assert retrieved is result


def test_update_missing_field_raises():
    """Calling ``update`` for an unregistered field raises ``KeyError``."""
    layout = FieldLayout()

    # Create field but do not add it to the layout
    field = DummyField(name="pressure")

    # Prepare dummy QIR
    result = QuadratureInterpolationResult(
        x=torch.empty(0), u=torch.empty(0), measure=torch.empty(0)
    )

    with pytest.raises(KeyError, match="No field named"):
        layout.update(field, result)


def test_getitem_missing_field_raises():
    """Accessing a non‑existent field via ``[]`` raises ``KeyError``."""
    layout = FieldLayout()

    with pytest.raises(KeyError, match="No field named"):
        _ = layout["nonexistent"]


def test_getitem_not_interpolated_raises():
    """Accessing a registered field that hasn't been interpolated raises ``RuntimeError``."""
    layout = FieldLayout()
    field = DummyField(name="strain")
    layout.add(field)

    with pytest.raises(RuntimeError, match="not yet interpolated"):
        _ = layout[field.name]
