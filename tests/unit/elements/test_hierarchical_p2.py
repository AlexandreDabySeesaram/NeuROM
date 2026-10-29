"""The non-nodal ``C``-folding path on a test-only hierarchical P2 triangle (T-WP2.5).

A hierarchical P2 basis (vertex hats + edge bubbles) spans the same space as the nodal P2 basis but
is not dual to the P2 DOFs. An element declared ``nodal=False`` must compute ``C = V^{-1}`` at
construction and fold it into the tabulation, recovering the nodal P2 basis exactly -- so the runtime
DOF transformation stays the identity and every downstream consumer sees a nodal basis.
"""

import pytest
import torch

from neurom.elements import (
    P2_TRIANGLE,
    HIERARCHICAL_P2_TRIANGLE,
    FiniteElement,
)
from neurom.shape_functions import HierarchicalP2Triangle, QuadraticTriangle

F64 = torch.float64


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# B_{s, a_s} = B_{s, b_s} = 1/2 for edge s = (a_s, b_s), edges (0,1),(1,2),(2,0).
_B = torch.tensor(
    [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]], dtype=F64
)


def test_duality_matrix_has_the_block_structure():
    """V = [[I, 0], [B, I]] -- vertex hats are nodal at vertices, bubbles unit at their midpoint.

    ``V`` is computed once at construction (the singleton is built at import, so its dtype follows
    the then-default); compare in float64, the entries (0, 1/2, 1) being exactly representable.
    """
    expected = torch.eye(6, dtype=F64)
    expected[3:, :3] = _B
    assert torch.allclose(HIERARCHICAL_P2_TRIANGLE.V.double(), expected, atol=1e-12)


def test_correction_C_is_the_inverse_block():
    """C = V^{-1} = [[I, 0], [-B, I]] folded into the tabulation (computed in float64)."""
    expected = torch.eye(6, dtype=F64)
    expected[3:, :3] = -_B
    assert HIERARCHICAL_P2_TRIANGLE.C is not None
    assert torch.allclose(HIERARCHICAL_P2_TRIANGLE.C.double(), expected, atol=1e-12)


def test_folded_basis_equals_nodal_p2():
    """The folded tabulation reproduces the nodal P2 basis at arbitrary points (to ~1e-14)."""
    elem = HIERARCHICAL_P2_TRIANGLE
    torch.manual_seed(0)
    # random reference points inside the triangle, shape (1, N_q, 2)
    r = torch.rand(1, 50, 2, dtype=F64)
    xi = r * (1.0 - r[..., [1, 0]])  # keep xi0 + xi1 < 1 (inside the simplex)
    folded = elem.N(xi)  # (1, N_q, 6) -- the FoldedBasis
    nodal = QuadraticTriangle().N(xi)  # (1, N_q, 6)
    assert torch.allclose(folded, nodal, atol=1e-13)


def test_the_coded_basis_is_not_nodal_so_folding_is_required():
    """Declaring the same hierarchical basis ``nodal=True`` is rejected: folding is not optional.

    This is the folding-omitted mutation at the unit level -- the coded basis is genuinely non-nodal
    (V != I), so an element that skipped the C-fold would silently mis-represent every quadratic.
    """
    with pytest.raises(ValueError, match="not dual"):
        FiniteElement(HierarchicalP2Triangle(), P2_TRIANGLE.dof_layout, nodal=True)
