"""Mathematical operations for FEM fields in the neurom package."""

from neurom.math.inner import inner
from neurom.math.integrate import integrate
from neurom.math.jacobian import jacobian
from neurom.math.second_derivative import second_derivative
from neurom.math.trace import trace
from neurom.math.identity import identity
from neurom.math.transpose import transpose

__all__ = [
    "inner",
    "integrate",
    "jacobian",
    "second_derivative",
    "trace",
    "identity",
    "transpose",
]
