"""Unit contract for :class:`~neurom.shape_functions.folded_basis.FoldedBasis`.

``FoldedBasis(coded, C)`` presents a coded reference basis :math:`\\hat\\Phi` as
:math:`\\hat H = \\hat\\Phi\\,C`. These tests isolate its contract directly (the hierarchical P2
element tests it only through a specific ``C``): the matmul composition and its orientation, the
inherited reference cell, the dtype cast at tabulation, and that derivatives fold through autograd.
"""

import torch

from neurom.shape_functions import LinearBar, LinearTriangle, FoldedBasis

F32 = torch.float32
F64 = torch.float64


def test_folds_basis_as_phi_times_C():
    """``N(xi) == coded.N(xi) @ C`` -- and an asymmetric C catches a transposed application."""
    coded = LinearTriangle()
    torch.manual_seed(0)
    C = torch.randn(3, 3, dtype=F64)  # asymmetric: Phi @ C != Phi @ C^T
    fb = FoldedBasis(coded, C)
    xi = torch.rand(1, 7, 2, dtype=F64)
    assert torch.allclose(fb.N(xi), coded.N(xi) @ C, atol=1e-14)


def test_inherits_the_reference_cell():
    """The folded basis lives on the same reference cell as the coded basis."""
    coded = LinearTriangle()
    fb = FoldedBasis(coded, torch.eye(3, dtype=F64))
    assert fb.reference_cell is coded.reference_cell


def test_casts_C_to_the_input_dtype():
    """C is held in float64 but tabulation follows the inputs (dtype/device-agnostic)."""
    coded = LinearTriangle()
    C = torch.randn(3, 3, dtype=F64)
    fb = FoldedBasis(coded, C)
    xi = torch.rand(1, 4, 2, dtype=F32)  # float32 inputs, float64 C
    out = fb.N(xi)
    assert out.dtype == F32
    assert torch.allclose(out, coded.N(xi) @ C.to(F32), atol=1e-6)


def test_derivatives_fold_through_autograd():
    """d/dxi of the folded basis equals (d/dxi of the coded basis) @ C, via autograd.

    LinearBar on [-1, 1] has N = [1/2(1-xi), 1/2(1+xi)], so dN/dxi = [-1/2, 1/2] everywhere.
    With C = [[1, 2], [3, 4]], the folded derivative is [-1/2, 1/2] @ C = [1, 1] at every point --
    an independent target the fold must reproduce without a hand-written derivative.
    """
    C = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=F64)
    fb = FoldedBasis(LinearBar(), C)
    xi = torch.linspace(-1.0, 1.0, 5, dtype=F64).reshape(1, 5, 1).requires_grad_(True)
    H = fb.N(xi)  # (1, 5, 2)
    grads = torch.stack(
        [torch.autograd.grad(H[..., a].sum(), xi, retain_graph=True)[0][..., 0]
         for a in range(2)],
        dim=-1,
    )  # (1, 5, 2)
    assert torch.allclose(grads, torch.ones_like(grads), atol=1e-12)
