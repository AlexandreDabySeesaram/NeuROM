"""Tests for the two engines ``apply`` (fast broadcast) and ``apply_pointwise`` (vmap).

Both dispatch over ``Sampling`` batch shapes with mixed-rank prefix-broadcast; ``apply``'s ``fn`` is
written with trailing-axis ops on full-batch tensors, ``apply_pointwise``'s ``fn`` on a single point.
"""

import pytest
import torch

from neurom.apply import apply, apply_pointwise
from neurom.samplings import ElementSampling, QuadratureSampling

F64 = torch.float64


def test_no_sampling_calls_fn_directly():
    """With no Sampling argument, apply just calls fn."""
    assert apply(lambda a, b: a + b, 2, 3) == 5


def test_same_rank_is_pointwise():
    """Two QuadratureSamplings of equal batch_shape -> point-wise op (the pre-existing path)."""
    a = QuadratureSampling(torch.randn(4, 3, 2, dtype=F64))
    b = QuadratureSampling(torch.randn(4, 3, 2, dtype=F64))
    res = apply(lambda u, v: u + v, a, b)
    assert isinstance(res, QuadratureSampling)
    assert res.batch_shape == (4, 3)
    assert torch.allclose(res.values, a.values + b.values)


def test_mixed_rank_broadcasts_element_over_quadrature():
    """Element-level (N_e,) operand broadcasts over the quadrature axis of a (N_e, N_q) operand.

    The fast fn operates on full-batch tensors (aligned to a common rank), so the matrix/vector
    contraction is written as a trailing-``...`` einsum; result == J_inv[e] @ x[e, q].
    """
    Ne, Nq, d = 4, 3, 2
    J_inv = ElementSampling(torch.randn(Ne, d, d, dtype=F64))
    x = QuadratureSampling(torch.randn(Ne, Nq, d, dtype=F64))
    res = apply(lambda Ji, xq: torch.einsum("...ij,...j->...i", Ji, xq), J_inv, x)
    assert isinstance(res, QuadratureSampling)  # richest subtype
    assert res.batch_shape == (Ne, Nq)
    expected = torch.einsum("eij,eqj->eqi", J_inv.values, x.values)
    assert torch.allclose(res.values, expected)


def test_constant_argument_is_broadcast():
    """A non-Sampling positional arg is a rank-0 constant, broadcast over every batch point."""
    x = QuadratureSampling(torch.randn(4, 3, 2, dtype=F64))
    c = torch.tensor([1.0, 2.0], dtype=F64)
    res = apply(lambda xq, cc: xq + cc, x, c)
    assert torch.allclose(res.values, x.values + c)


def test_non_prefix_batch_shape_raises():
    """A Sampling whose batch_shape is not a prefix of the richest one is rejected."""
    with pytest.raises(ValueError, match="prefix"):
        apply(
            lambda Ji, xq: Ji @ xq,
            ElementSampling(torch.randn(5, 2, 2, dtype=F64)),   # N_e = 5
            QuadratureSampling(torch.randn(4, 3, 2, dtype=F64)),  # N_e = 4 -> mismatch
        )


def test_kwargs_are_forwarded_unbatched():
    """Keyword arguments reach fn unbatched (as the docstring claims)."""
    x = QuadratureSampling(torch.randn(4, 3, 2, dtype=F64))
    res = apply(lambda xq, *, k: xq * k, x, k=3.0)
    assert torch.allclose(res.values, x.values * 3.0)


def test_three_axis_broadcast():
    """Ranks 1/2/3 compose: each lower-rank operand broadcasts over the axes it lacks.

    Richest batch (M, N_e, N_q); a (M, N_e) operand broadcasts over N_q; a (M,) operand over both
    N_e and N_q. This is the 'free batch axes' claim beyond element+quadrature (e.g. a future mode M).
    """
    M, Ne, Nq, d = 2, 4, 3, 2

    class _Rank1(ElementSampling):
        @property
        def batch_shape(self):
            return self.shape[:1]

    class _Rank2(ElementSampling):
        @property
        def batch_shape(self):
            return self.shape[:2]

    class _Rank3(ElementSampling):
        @property
        def batch_shape(self):
            return self.shape[:3]

    low = _Rank1(torch.randn(M, d, dtype=F64))          # (M,) batch
    mid = _Rank2(torch.randn(M, Ne, d, dtype=F64))      # (M, N_e) batch
    high = _Rank3(torch.randn(M, Ne, Nq, d, dtype=F64))  # (M, N_e, N_q) batch

    res = apply(lambda a, b, c: a + b + c, low, mid, high)
    assert res.batch_shape == (M, Ne, Nq)
    expected = (
        low.values[:, None, None, :]
        + mid.values[:, :, None, :]
        + high.values
    )
    assert torch.allclose(res.values, expected)


# --- apply_pointwise (vmap engine) ------------------------------------------------------------------
def test_apply_pointwise_vmaps_a_single_point_fn():
    """A fn written for ONE matrix (bare ``.diagonal()``) only works under the vmap engine."""
    m = QuadratureSampling(torch.randn(3, 2, 4, 4, dtype=F64))  # (N_e, N_q, 4, 4)
    res = apply_pointwise(lambda a: a.diagonal().sum().reshape(1), m)
    assert res.batch_shape == (3, 2)
    expected = torch.diagonal(m.values, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    assert torch.allclose(res.values, expected)


def test_apply_pointwise_mixed_rank():
    """vmap de-batches operands to single points, so ``@`` works and element broadcasts over quad."""
    Ne, Nq, d = 4, 3, 2
    J_inv = ElementSampling(torch.randn(Ne, d, d, dtype=F64))
    x = QuadratureSampling(torch.randn(Ne, Nq, d, dtype=F64))
    res = apply_pointwise(lambda Ji, xq: Ji @ xq, J_inv, x)
    assert res.batch_shape == (Ne, Nq)
    assert torch.allclose(res.values, torch.einsum("eij,eqj->eqi", J_inv.values, x.values))
