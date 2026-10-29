"""Concrete finite elements as ready-to-use singletons.

Each element is a pure reference-space object (no geometry, no trainable state), so a single
shared instance serves every mesh and dtype -- hence module-level constants rather than
factory functions. Vector-valued elements are a scalar element wrapped by
:func:`VectorElement`; ``DG0`` needs a cell, so it keeps a small factory alongside the
per-cell constants.
"""

from neurom.elements.finite_element import FiniteElement, DoF, VALUE, D1
from neurom.meshes.topology import EntityDim
from neurom.reference_cells import ReferenceCell, BAR, TRIANGLE
from neurom.shape_functions import (
    LinearBar,
    LinearTriangle,
    QuadraticTriangle,
    HierarchicalP2Triangle,
    HermiteBeam,
    Constant,
)


def VectorElement(base: FiniteElement, dim: int) -> FiniteElement:
    """A vector-valued element: ``base``'s scalar layout replicated across ``dim`` components.

    The reference basis and DOF layout are shared with ``base``; only ``value_shape`` changes to
    ``(dim,)``, so each scalar DOF becomes ``dim`` DOFs (one per component).

    Args:
        base (FiniteElement): The scalar element to vectorise (e.g. :data:`P1_TRIANGLE`).
        dim (int): Number of value components.

    Returns:
        FiniteElement: Same reference basis and DOF layout as ``base``, with ``value_shape=(dim,)``.
    """
    return FiniteElement(
        base.reference_basis,
        base.dof_layout,
        value_shape=(dim,),
        pullback=base.pullback,
    )


def DG0(cell: ReferenceCell, value_shape=()) -> FiniteElement:
    r"""Piecewise-constant (:math:`P_0`) element: a single cell-value DOF (dimension-agnostic).

    One DOF, the constant cell value -- basis :math:`N_0\equiv1`, functional :math:`\ell(v)=v|_K`
    -- attached to the top-dimensional cell entity (``EDGE`` for a bar, ``FACE`` for a triangle).
    """
    cell_dim = EntityDim(cell.simplex.shape[1])  # EDGE for a bar, FACE for a triangle
    return FiniteElement(
        Constant(cell),
        [DoF(entity_dim=cell_dim, local_index=0, kind=VALUE)],
        value_shape=value_shape,
    )


P1_BAR = FiniteElement(
    LinearBar(),
    [
        DoF(entity_dim=EntityDim.VERTEX, local_index=0, kind=VALUE),
        DoF(entity_dim=EntityDim.VERTEX, local_index=1, kind=VALUE),
    ],
)
r"""**P1 (linear Lagrange) on the bar.** Reference cell :math:`\hat K=[-1,1]`, space :math:`P_1`.
Nodal basis :math:`N_i(\xi)=\tfrac12(1\mp\xi)` -- the barycentric coordinates
:math:`(\lambda_0,\lambda_1)`. DOFs: the value at each vertex, :math:`\ell_i(v)=v(\xi_i)` with
:math:`\xi_0=-1,\ \xi_1=+1`. Nodal basis, so the DOF transformation :math:`M_e` is the identity."""


P1_TRIANGLE = FiniteElement(
    LinearTriangle(),
    [DoF(entity_dim=EntityDim.VERTEX, local_index=i, kind=VALUE) for i in range(3)],
)
r"""**P1 (linear Lagrange) on the triangle.** Reference cell the unit triangle
:math:`\{\lambda_i\ge0,\ \sum_i\lambda_i=1\}`, space :math:`P_1`. Nodal basis equal to the
barycentric coordinates, :math:`N_i=\lambda_i\ (i=0,1,2)`. DOFs: the value at each of the three
vertices, :math:`\ell_i(v)=v(\mathbf x_i)`. Identity DOF transformation."""


P2_TRIANGLE = FiniteElement(
    QuadraticTriangle(),
    [
        DoF(entity_dim=EntityDim.VERTEX, local_index=0, kind=VALUE),
        DoF(entity_dim=EntityDim.VERTEX, local_index=1, kind=VALUE),
        DoF(entity_dim=EntityDim.VERTEX, local_index=2, kind=VALUE),
        DoF(entity_dim=EntityDim.EDGE, local_index=0, kind=VALUE),
        DoF(entity_dim=EntityDim.EDGE, local_index=1, kind=VALUE),
        DoF(entity_dim=EntityDim.EDGE, local_index=2, kind=VALUE),
    ],
)
r"""**P2 (quadratic Lagrange) on the triangle.** Space :math:`P_2`, 6 DOFs. Nodal basis in
barycentric coordinates: vertices :math:`N_{v_i}=\lambda_i(2\lambda_i-1)` and edge midpoints
:math:`N_{e_{ij}}=4\lambda_i\lambda_j`, edges ordered :math:`(0,1),(1,2),(2,0)`. DOFs: the value
at the three vertices and the three edge midpoints. Identity DOF transformation."""


HIERARCHICAL_P2_TRIANGLE = FiniteElement(
    HierarchicalP2Triangle(), P2_TRIANGLE.dof_layout, nodal=False
)
r"""**Non-nodal P2 on the triangle** -- same :math:`P_2` space and nodal P2 DOFs as
:data:`P2_TRIANGLE`, but a *hierarchical* coded basis (three linear vertex hats
:math:`\lambda_i` plus three quadratic edge bubbles :math:`4\lambda_i\lambda_j`) that is **not**
dual to the DOFs. Declared ``nodal=False``, so construction computes :math:`C = V^{-1}` and folds
it into the tabulation (see :class:`~neurom.shape_functions.folded_basis.FoldedBasis`), recovering
the nodal P2 basis exactly. Not a production element: it exists to exercise and verify the
non-nodal :math:`C`-folding path (every other library element is nodal, so its coded basis equals
the basis it presents)."""


HERMITE = FiniteElement(
    HermiteBeam(),
    [
        DoF(entity_dim=EntityDim.VERTEX, local_index=0, kind=VALUE),
        DoF(entity_dim=EntityDim.VERTEX, local_index=0, kind=D1),
        DoF(entity_dim=EntityDim.VERTEX, local_index=1, kind=VALUE),
        DoF(entity_dim=EntityDim.VERTEX, local_index=1, kind=D1),
    ],
)
r"""**Cubic Hermite beam on the bar.** Reference cell :math:`\hat K=[-1,1]`, space :math:`P_3`,
4 DOFs. Reference basis :math:`H_1=\tfrac14(1-\xi)^2(2+\xi)`,
:math:`H_2=\tfrac14(1-\xi)^2(\xi+1)`, :math:`H_3=\tfrac14(1+\xi)^2(2-\xi)`,
:math:`H_4=\tfrac14(1+\xi)^2(\xi-1)`. DOFs at each vertex are the value and the **physical**
first derivative, :math:`(\,v\mapsto v(x_i),\ v\mapsto v'(x_i)\,)`, at :math:`\xi_0=-1` then
:math:`\xi_1=+1`. The derivative DOFs are physical, so :math:`M_e` scales with the cell size
(geometry-dependent, unlike the nodal Lagrange elements)."""


DG0_BAR = DG0(BAR)
r"""**DG0 (piecewise constant) on the bar.** Space :math:`P_0`, a single cell-value DOF
(:math:`N_0\equiv1`, :math:`\ell(v)=v|_K`). See :func:`DG0`."""

DG0_TRIANGLE = DG0(TRIANGLE)
r"""**DG0 (piecewise constant) on the triangle.** Space :math:`P_0`, a single cell-value DOF
(:math:`N_0\equiv1`, :math:`\ell(v)=v|_K`). See :func:`DG0`."""
