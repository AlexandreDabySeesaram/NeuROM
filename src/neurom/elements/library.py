"""Concrete finite elements as ready-to-use singletons.

Each element is a pure reference-space object (no geometry, no trainable state), so a single
shared instance serves every mesh and dtype -- hence module-level constants rather than
factory functions. Vector-valued elements are a scalar element wrapped by
:func:`VectorElement`; ``DG0`` needs a cell, so it keeps a small factory alongside the
per-cell constants.
"""

from neurom.elements.finite_element import FiniteElement, VALUE, D1
from neurom.meshes.topology import EntityDim
from neurom.reference_elements import ReferenceElement, Bar, Triangle
from neurom.shape_functions import (
    LinearBar,
    LinearTriangle,
    QuadraticTriangle,
    HermiteBeam,
    Constant,
)


def VectorElement(base: FiniteElement, dim: int) -> FiniteElement:
    """A vector-valued element: ``base``'s scalar layout with ``dim`` components.

    Args:
        base (FiniteElement): The scalar element to vectorise (e.g. :data:`P1_TRIANGLE`).
        dim (int): Number of value components.

    Returns:
        FiniteElement: Same reference basis and DOF layout as ``base``, with
        ``value_shape=(dim,)``.
    """
    return FiniteElement(
        base.reference_basis,
        base.dof_layout,
        value_shape=(dim,),
        pullback=base.pullback,
    )


def DG0(cell: ReferenceElement, value_shape=()) -> FiniteElement:
    """Piecewise-constant element: a single value DOF on the cell (dimension-agnostic)."""
    cell_dim = EntityDim(cell.simplex.shape[1])  # EDGE for a bar, FACE for a triangle
    return FiniteElement(
        Constant(cell), [(cell_dim, 0, VALUE)], value_shape=value_shape
    )


#: Linear Lagrange on the bar: one value DOF per vertex.
P1_BAR = FiniteElement(
    LinearBar(), [(EntityDim.VERTEX, 0, VALUE), (EntityDim.VERTEX, 1, VALUE)]
)

#: Linear Lagrange on the triangle: one value DOF per vertex.
P1_TRIANGLE = FiniteElement(
    LinearTriangle(), [(EntityDim.VERTEX, i, VALUE) for i in range(3)]
)

#: Quadratic Lagrange on the triangle: a value DOF at each vertex and at each edge midpoint.
P2_TRIANGLE = FiniteElement(
    QuadraticTriangle(),
    [
        (EntityDim.VERTEX, 0, VALUE),
        (EntityDim.VERTEX, 1, VALUE),
        (EntityDim.VERTEX, 2, VALUE),
        (EntityDim.EDGE, 0, VALUE),
        (EntityDim.EDGE, 1, VALUE),
        (EntityDim.EDGE, 2, VALUE),
    ],
)

#: Cubic Hermite beam: a value and a first-derivative DOF at each of the two vertices.
HERMITE = FiniteElement(
    HermiteBeam(),
    [
        (EntityDim.VERTEX, 0, VALUE),
        (EntityDim.VERTEX, 0, D1),
        (EntityDim.VERTEX, 1, VALUE),
        (EntityDim.VERTEX, 1, D1),
    ],
)

#: Piecewise-constant on the bar / triangle (the common :func:`DG0` cells).
DG0_BAR = DG0(Bar())
DG0_TRIANGLE = DG0(Triangle())
