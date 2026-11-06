"""Physical-to-reference DOF transformations.

One ``Transformation`` per (element, mapping): it assembles ``L_e`` from the element's
per-kind **functionals** and inverts it to ``M_e`` (identity short-circuit when ``L_e≡I``).
The functionals are the only extension point (and are reused by ``FunctionSpace.interpolate``).
"""

from neurom.dof_transformations.transformation import Transformation
from neurom.dof_transformations.reference_to_physical import (
    reference_to_physical,
    reference_to_physical_at,
)
from neurom.dof_transformations.functionals import (
    DofFunctional,
    ValueFunctional,
    GradientFunctional,
    FUNCTIONALS,
)

__all__ = [
    "Transformation",
    "reference_to_physical",
    "reference_to_physical_at",
    "DofFunctional",
    "ValueFunctional",
    "GradientFunctional",
    "FUNCTIONALS",
]
