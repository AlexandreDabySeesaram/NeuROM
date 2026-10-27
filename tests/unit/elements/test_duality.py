"""Construction-time duality / unisolvence check (C2).

Every library element's reference basis must be dual to its DOFs -- the duality matrix
``V_{ab} = l_a(phi_b)`` equals the identity -- and a mis-declared nodal element (basis not dual,
or wrong number of basis functions) must be rejected at construction with an explicit message.
"""

import pytest
import torch

from neurom.elements import (
    FiniteElement,
    DoF,
    VALUE,
    EntityDim,
    P1_BAR,
    P1_TRIANGLE,
    P2_TRIANGLE,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
    VectorElement,
)
from neurom.shape_functions.shape_function import ShapeFunction
from neurom.shape_functions.quadratic_bar import QuadraticBar
from neurom.reference_cells import BAR

LIBRARY = [
    P1_BAR,
    P1_TRIANGLE,
    P2_TRIANGLE,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
    VectorElement(P1_TRIANGLE, 2),
]
IDS = [
    "P1_BAR",
    "P1_TRIANGLE",
    "P2_TRIANGLE",
    "HERMITE",
    "DG0_BAR",
    "DG0_TRIANGLE",
    "vector",
]


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


class _MonomialBar(ShapeFunction):
    """Monomial basis ``{1, xi}`` on the bar: spans P1 but is NOT nodal at the vertices."""

    def __init__(self):
        super().__init__(BAR)

    def N(self, xi):
        x = xi[..., 0]
        return torch.stack([torch.ones_like(x), x], dim=-1)  # (..., 2)


@pytest.mark.parametrize("element", LIBRARY, ids=IDS)
def test_duality_matrix_is_identity_for_library_elements(element):
    """``V_{ab} = l_a(phi_b) == I`` for every library element (T-WP2.1).

    The reference functional of each DOF applied to the coded reference basis at that DOF's
    reference point reproduces the Kronecker delta -- i.e. the hand-written basis really is the
    nodal basis dual to the declared DOFs. Run in float64, checked to round-off.
    """
    V = element.duality_matrix()
    assert V is not None
    assert torch.allclose(V, torch.eye(element.n_dofs, dtype=V.dtype), atol=1e-12)


def test_nodal_element_with_non_dual_basis_is_rejected():
    """A nodal element whose basis is not dual to its DOFs is rejected at construction (T-WP2.6).

    The monomial basis ``{1, xi}`` spans P1 but is not nodal at the vertices, so ``V != I``; the
    element is declared nodal (the default), so construction must raise.
    """
    with pytest.raises(ValueError, match="not dual"):
        FiniteElement(
            _MonomialBar(),
            [DoF(EntityDim.VERTEX, 0, VALUE), DoF(EntityDim.VERTEX, 1, VALUE)],
        )


def test_non_unisolvent_element_is_rejected():
    """A nodal element with ``n_dofs != n_ref`` is rejected at construction (T-WP2.6).

    Two vertex-value DOFs against a quadratic (3-function) basis is not unisolvent.
    """
    with pytest.raises(ValueError, match="unisolvent"):
        FiniteElement(
            QuadraticBar(),
            [DoF(EntityDim.VERTEX, 0, VALUE), DoF(EntityDim.VERTEX, 1, VALUE)],
        )
