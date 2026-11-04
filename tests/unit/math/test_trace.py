import pytest
import torch

# Import library modules
from neurom.math.trace import trace, trace_point
from neurom.samplings import QuadratureSampling

torch.set_default_dtype(torch.float32)


class TestTracePoint:
    """Test trace_point() acting on a single (non-batched) field tensor.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_scalar_returns_clone(self):
        """A 1D (scalar) field is returned unchanged as a fresh clone."""
        u = torch.tensor([7.0])
        result = trace_point(u)

        assert result.shape == (1,)
        assert result is not u
        assert result == pytest.approx(u, rel=self.relative_tolerance)

    def test_square_matrix(self):
        """The trace of a square matrix is the sum of its diagonal, kept as shape (1,)."""
        u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = trace_point(u)

        assert result.shape == (1,)
        assert result == pytest.approx(torch.tensor([5.0]), rel=self.relative_tolerance)

    def test_identity_trace_equals_dimension(self):
        """The trace of the d x d identity is d, kept as shape (1,)."""
        u = torch.eye(4)
        result = trace_point(u)

        assert result.shape == (1,)
        assert result == pytest.approx(torch.tensor([4.0]), rel=self.relative_tolerance)


class TestTrace:
    """Test trace() acting on a Sampling over its (N_e, N_q) batch.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_matrix_sampling(self):
        """trace() reduces a (N_e, N_q, d, d) field to a per-point scalar."""
        # Build a batch of matrices with known diagonals.
        base = torch.tensor([[1.0, 9.0], [9.0, 5.0]])  # trace = 6
        values = base.expand(2, 3, 2, 2).clone()
        s = QuadratureSampling(values)

        result = trace(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (2, 3)
        # trace_point reduces (d, d) -> (1,), so the field collapses to a scalar.
        assert result.f_shape == (1,)
        assert result.values == pytest.approx(
            torch.full((2, 3, 1), 6.0), rel=self.relative_tolerance
        )

    def test_trace_of_identity_sampling(self):
        """The trace of an identity field equals its dimension everywhere."""
        values = torch.eye(3).expand(2, 4, 3, 3).clone()
        s = QuadratureSampling(values)

        result = trace(s)

        assert result.f_shape == (1,)
        assert result.values == pytest.approx(
            torch.full((2, 4, 1), 3.0), rel=self.relative_tolerance
        )
