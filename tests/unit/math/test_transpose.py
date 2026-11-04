import pytest
import torch

# Import library modules
from neurom.math.transpose import transpose, transpose_point
from neurom.samplings import QuadratureSampling

torch.set_default_dtype(torch.float32)


class TestTransposePoint:
    """Test transpose_point() acting on a single (non-batched) field tensor.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_scalar_returns_clone(self):
        """A 1D tensor (scalar/vector) is returned unchanged as a fresh clone."""
        u = torch.tensor([3.0])
        result = transpose_point(u)

        assert result.shape == (1,)
        assert result is not u
        assert result == pytest.approx(u, rel=self.relative_tolerance)

    def test_vector_returns_clone(self):
        """A 1D vector is returned unchanged as a fresh clone."""
        u = torch.tensor([1.0, 2.0, 3.0])
        result = transpose_point(u)

        assert result.shape == (3,)
        assert result is not u
        assert result == pytest.approx(u, rel=self.relative_tolerance)

    def test_square_matrix_is_transposed(self):
        """A square matrix has its last two dimensions swapped."""
        u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = transpose_point(u)

        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert result.shape == (2, 2)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_transpose_is_involution(self):
        """Transposing a matrix field twice recovers the original."""
        u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = transpose_point(transpose_point(u))

        assert result == pytest.approx(u, rel=self.relative_tolerance)

    def test_higher_order_cubic_tensor_is_transposed(self):
        """A (d, d, d) tensor has its last two axes swapped."""
        u = torch.arange(27.0).reshape(3, 3, 3)
        result = transpose_point(u)

        assert result.shape == (3, 3, 3)
        assert result == pytest.approx(u.transpose(-1, -2), rel=self.relative_tolerance)

    def test_non_square_matrix_raises(self):
        """A tensor whose shape entries differ is rejected."""
        u = torch.ones(2, 3)
        with pytest.raises(ValueError):
            _ = transpose_point(u)


class TestTranspose:
    """Test transpose() acting on a Sampling over its (N_e, N_q) batch.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_scalar_returns_clone(self):
        """A scalar field (N_e, N_q, 1) is returned unchanged."""
        s = QuadratureSampling(torch.arange(6.0).reshape(3, 2, 1))
        result = transpose(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (3, 2)
        assert result.f_shape == (1,)
        assert result.values == pytest.approx(s.values, rel=self.relative_tolerance)

    def test_vector_returns_clone(self):
        """A vector field (N_e, N_q, d) has no meaningful transpose -> unchanged."""
        s = QuadratureSampling(torch.arange(12.0).reshape(2, 2, 3))
        result = transpose(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (2, 2)
        assert result.f_shape == (3,)
        assert result.values == pytest.approx(s.values, rel=self.relative_tolerance)

    def test_matrix_is_transposed(self):
        """A matrix field (N_e, N_q, d, d) has its last two dims swapped."""
        s = QuadratureSampling(torch.arange(2 * 2 * 2 * 2.0).reshape(2, 2, 2, 2))
        result = transpose(s)

        assert isinstance(result, QuadratureSampling)
        assert result.batch_shape == (2, 2)
        assert result.f_shape == (2, 2)
        assert result.values == pytest.approx(
            s.values.transpose(-1, -2), rel=self.relative_tolerance
        )

    def test_transpose_is_involution(self):
        """Transposing a matrix field twice recovers the original."""
        s = QuadratureSampling(torch.randn(3, 4, 3, 3))
        result = transpose(transpose(s))

        assert result.values == pytest.approx(s.values, rel=self.relative_tolerance)

    def test_higher_order_tensor_is_transposed(self):
        """A higher-order tensor field swaps its last two axes."""
        s = QuadratureSampling(torch.randn(2, 3, 2, 2, 2))
        result = transpose(s)

        assert result.f_shape == (2, 2, 2)
        assert result.values == pytest.approx(
            s.values.transpose(-1, -2), rel=self.relative_tolerance
        )
