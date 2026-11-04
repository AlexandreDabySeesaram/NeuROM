import pytest
import torch

# Import library modules
from neurom.math.identity import identity, identity_point
from neurom.samplings import QuadratureSampling

torch.set_default_dtype(torch.float32)


class TestIdentityPoint:
    """Test identity_point() acting on a single (non-batched) field tensor.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_scalar(self):
        """For a scalar (1,) field the identity is one."""
        u = torch.tensor([42.0])
        result = identity_point(u)

        expected = torch.ones(1)
        assert result.shape == (1,)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_square_matrix(self):
        """For a square matrix (d, d) the identity is the d x d identity matrix."""
        u = torch.arange(9.0).reshape(3, 3)
        result = identity_point(u)

        expected = torch.eye(3)
        assert result.shape == (3, 3)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_preserves_dtype_and_device(self):
        """The identity inherits dtype/device from the input tensor."""
        u = torch.zeros(2, 2, dtype=torch.float64)
        result = identity_point(u)

        assert result.dtype == torch.float64
        assert result.device == u.device

    def test_vector_raises(self):
        """A non-scalar, non-square field has no defined identity."""
        u = torch.ones(3)
        with pytest.raises(ValueError):
            _ = identity_point(u)

    def test_non_square_matrix_raises(self):
        """A rectangular matrix has no defined identity."""
        u = torch.ones(2, 3)
        with pytest.raises(ValueError):
            _ = identity_point(u)


class TestIdentity:
    """Test identity() acting on a Sampling, expanded over its batch_shape.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_matrix_sampling(self):
        """identity() expands the d x d identity over the (N_e, N_q) batch."""
        s = QuadratureSampling(torch.randn(2, 3, 2, 2))
        result = identity(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (2, 3)
        assert result.f_shape == (2, 2)

        expected = torch.eye(2).expand(2, 3, 2, 2)
        assert result.values == pytest.approx(expected, rel=self.relative_tolerance)

    def test_scalar_sampling(self):
        """identity() of a scalar field is ones over the batch."""
        s = QuadratureSampling(torch.randn(2, 3, 1))
        result = identity(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (2, 3)
        assert result.f_shape == (1,)
        assert result.values == pytest.approx(
            torch.ones(2, 3, 1), rel=self.relative_tolerance
        )
