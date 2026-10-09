"""Finite elements: reference basis + declarative DOF layout (DOF counts deduced)."""

from neurom.elements.finite_element import (
    FiniteElement,
    VALUE,
    D1,
    FLUX,
    NORMAL_DERIVATIVE,
)
from neurom.meshes.topology import EntityDim
from neurom.elements.library import (
    VectorElement,
    DG0,
    P1_BAR,
    P1_TRIANGLE,
    P2_TRIANGLE,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
)

__all__ = [
    "FiniteElement",
    "EntityDim",
    "VALUE",
    "D1",
    "FLUX",
    "NORMAL_DERIVATIVE",
    "VectorElement",
    "DG0",
    "P1_BAR",
    "P1_TRIANGLE",
    "P2_TRIANGLE",
    "HERMITE",
    "DG0_BAR",
    "DG0_TRIANGLE",
]
