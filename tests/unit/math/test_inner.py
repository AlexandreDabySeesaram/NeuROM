import pytest
import torch

# Import library modules
from neurom.inner import inner

torch.set_default_dtype(torch.float32)


class TestInner:
    """Test inner() computation for fields R^d → R^m.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_incompatible_dimensions(self):
        """Check we throw for incompatible dimensions between the two fields we want to compute inner product of"""

        # Different number of elements
        u = torch.ones(3, 2, 1)
        v = torch.ones(2, 2, 1)

        with pytest.raises(ValueError):
            _ = inner(u, v)

        # Different number of quadrature points
        v = torch.ones(3, 1, 1)

        with pytest.raises(ValueError):
            _ = inner(u, v)

        # Different dimension
        v = torch.ones(3, 1, 2)

        with pytest.raises(ValueError):
            _ = inner(u, v)

    def test_scalar(self):
        """Compute inner product for a scalar field.

        Choose: u(x) = x**2
        Expected: u(x) * v(x) = x**3
        """
        x = torch.tensor([[-3.0, 7.0], [10.0, 5.0], [6.0, -9.0]]).unsqueeze(
            -1
        )  # (3, 2, 1)
        u = x**2
        v = x.clone()

        assert u.shape == (3, 2, 1)
        assert v.shape == (3, 2, 1)

        result = inner(u, v)
        expected = x**3
        assert expected.shape == (3, 2, 1)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_scalar_one_quadrature_point(self):
        """Compute inner product for a scalar field.

        Check behavior for one quadrature points that the quadrature dimension is not suqeezed accidently.

        Choose: u(x) = x**2
        Expected: u(x) * v(x) = x**3
        """
        x = torch.tensor([-3.0, 7.0, 10.0]).reshape(3, 1, 1)  # (3, 1, 1)
        u = x**2
        v = x.clone()

        assert u.shape == (3, 1, 1)
        assert v.shape == (3, 1, 1)

        result = inner(u, v)
        expected = x**3
        assert expected.shape == (3, 1, 1)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_vector(self):
        """Inner product for a 2‑component vector field defined as functions of x.

        We construct a scalar field ``x`` of shape (2, 2, 1) and define
        ``u`` and ``v`` component‑wise as simple functions of ``x``. The analytic
        inner product can then be expressed directly in terms of ``x``.
        """
        # Create a scalar field x with shape (3, 2)
        x = torch.arange(0.0, 4.0).view(2, 2)  # [[0,1],[2,3]]

        # Define vector fields u and v as functions of x
        # u = [x, x**2]
        # v = [2*x, 3*x**2]
        u = torch.stack([x, x**2], dim=-1)  # (3, 2, 2)
        v = torch.stack([2 * x, 3 * x**2], dim=-1)  # (3, 2, 2)

        assert u.shape == (2, 2, 2)
        assert v.shape == (2, 2, 2)

        # Analytic expected inner product:
        #   sum_i u_i * v_i = 2*x**2 + 3*x**4
        expected = (2 * x**2 + 3 * x**4).unsqueeze(-1)  # (3, 2, 1)
        assert expected.shape == (2, 2, 1)

        result = inner(u, v)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)

    def test_tensor(self):
        """Inner product for a 3×3 tensor field defined as functions of x.

        We create a scalar field ``x`` of shape (2, 2, 1) and expand it to a
        full 3×3 tensor field by broadcasting over the identity matrix. The
        analytic inner product is then straightforward to compute.
        """
        # Scalar field x with shape (2, 2)
        x = torch.arange(1.0, 5.0).view(2, 2)  # [[1,2],[3,4]]

        # Identity matrix of size 3
        I = torch.eye(3).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

        # Define tensor fields u and v as scaled identity matrices
        # u = x * I, v = 2 * x * I
        u = x.unsqueeze(-1).unsqueeze(-1) * I  # broadcasts to (2,2,3,3)
        v = 2 * x.unsqueeze(-1).unsqueeze(-1) * I  # (2,2,3,3)

        assert u.shape == (2, 2, 3, 3)
        assert v.shape == (2, 2, 3, 3)

        # Analytic expected inner product:
        # only diagonal entries are non‑zero, each equals x and 2*x:
        #   sum_{i,j} u_ij * v_ij = 3 * (x * 2*x) = 6 * x**2
        expected = (6 * x**2).unsqueeze(-1)  # (2, 2, 1)
        assert expected.shape == (2, 2, 1)
        result = inner(u, v)
        assert result == pytest.approx(expected, rel=self.relative_tolerance)
