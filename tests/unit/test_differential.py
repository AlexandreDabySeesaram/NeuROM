import pytest
import torch

# Import library modules
from neurom import differential

torch.set_default_dtype(torch.float32)


class TestJacobianField:
    """Test jacobian_field computation for fields R^d -> R^m.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    # ------------------------------------------------------------------ #
    # R^1 -> R^1  (scalar field in 1D space)                              #
    # ------------------------------------------------------------------ #

    def test_scalar_1d_space_exp(self):
        """u = exp(x), du/dx = exp(x). Shape: x (3,2,1), u (3,2,1), J (3,2,1,1)"""
        x = torch.tensor(
            [[-3.0, 7.0], [10.0, 5.0], [6.0, -9.0]], requires_grad=True
        ).unsqueeze(-1)  # (3, 2, 1)
        u = torch.exp(x)  # (3, 2, 1)
        grad = differential.jacobian_field(x, u)  # (3, 2, 1, 1)

        assert x.shape == (3, 2, 1)
        assert u.shape == (3, 2, 1)
        assert grad.shape == (3, 2, 1, 1)
        assert grad.detach() == pytest.approx(
            u.unsqueeze(-1).detach(), self.relative_tolerance
        )

    def test_scalar_1d_space_polynomial(self):
        """u = x^3, du/dx = 3x^2. Shape: x (2,3,1), u (2,3,1), J (2,3,1,1)"""
        x = torch.linspace(-2.0, 2.0, 6).reshape(2, 3, 1).requires_grad_(True)
        u = x**3
        grad = differential.jacobian_field(x, u)

        assert x.shape == (2, 3, 1)
        assert u.shape == (2, 3, 1)
        assert grad.shape == (2, 3, 1, 1)

        expected = (3 * x**2).unsqueeze(-1).detach()
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^2 -> R^1  (scalar field in 2D space)                              #
    # ------------------------------------------------------------------ #

    def test_scalar_2d_space_linear(self):
        """u = 2*x0 + 3*x1, grad = [2, 3]. Shape: x (2,2,2), u (2,2,1), J (2,2,1,2)"""
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )  # (2, 2, 2)
        u = 2.0 * x[..., :1] + 3.0 * x[..., 1:]  # (2, 2, 1)
        grad = differential.jacobian_field(x, u)  # (2, 2, 1, 2)

        assert x.shape == (2, 2, 2)
        assert u.shape == (2, 2, 1)
        assert grad.shape == (2, 2, 1, 2)

        expected = torch.zeros(2, 2, 1, 2)
        expected[..., 0, 0] = 2.0
        expected[..., 0, 1] = 3.0
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    def test_scalar_2d_space_nonlinear(self):
        """u = x0^2 * x1, grad = [2*x0*x1, x0^2]. Shape: x (1,2,2), u (1,2,1), J (1,2,1,2)"""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
        u = (x[..., :1] ** 2) * x[..., 1:]  # (1, 2, 1)
        grad = differential.jacobian_field(x, u)  # (1, 2, 1, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 1)
        assert grad.shape == (1, 2, 1, 2)

        x0 = x[..., 0:1].detach()
        x1 = x[..., 1:2].detach()
        expected = torch.cat([2 * x0 * x1, x0**2], dim=-1).unsqueeze(-2)  # (1, 2, 1, 2)
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    def test_scalar_2d_space_exp_sin(self):
        """u = exp(x0)*sin(x1). Shape: x (1,2,2), u (1,2,1), J (1,2,1,2)"""
        x = torch.tensor([[[0.5, 1.0], [1.0, 0.5]]], requires_grad=True)  # (1, 2, 2)
        u = torch.exp(x[..., :1]) * torch.sin(x[..., 1:])  # (1, 2, 1)
        grad = differential.jacobian_field(x, u)  # (1, 2, 1, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 1)
        assert grad.shape == (1, 2, 1, 2)

        x0 = x[..., 0:1].detach()
        x1 = x[..., 1:2].detach()
        du_dx0 = torch.exp(x0) * torch.sin(x1)
        du_dx1 = torch.exp(x0) * torch.cos(x1)
        expected = torch.cat([du_dx0, du_dx1], dim=-1).unsqueeze(-2)  # (1, 2, 1, 2)
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^2 -> R^2  (vector field in 2D space)                              #
    # ------------------------------------------------------------------ #

    def test_vector_2d_space_linear(self):
        """
        u = [2*x0 + x1, x0 - 3*x1], Jacobian = [[2, 1], [1, -3]].
        Shape: x (2,2,2), u (2,2,2), J (2,2,2,2)
        """
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [-1.0, 2.0]]], requires_grad=True
        )  # (2, 2, 2)
        u = torch.cat(
            [2.0 * x[..., :1] + x[..., 1:], x[..., :1] - 3.0 * x[..., 1:]], dim=-1
        )  # (2, 2, 2)
        grad = differential.jacobian_field(x, u)  # (2, 2, 2, 2)

        assert x.shape == (2, 2, 2)
        assert u.shape == (2, 2, 2)
        assert grad.shape == (2, 2, 2, 2)

        expected = torch.tensor([[2.0, 1.0], [1.0, -3.0]]).expand(2, 2, 2, 2)
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    def test_vector_2d_space_nonlinear(self):
        """
        u = [x0^2, x0*x1], Jacobian = [[2*x0, 0], [x1, x0]].
        Shape: x (1,2,2), u (1,2,2), J (1,2,2,2)
        """
        x = torch.tensor([[[1.0, 2.0], [3.0, 0.5]]], requires_grad=True)  # (1, 2, 2)
        u = torch.stack([x[..., 0] ** 2, x[..., 0] * x[..., 1]], dim=-1)  # (1, 2, 2)
        grad = differential.jacobian_field(x, u)  # (1, 2, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 2)
        assert grad.shape == (1, 2, 2, 2)

        x0 = x[..., 0].detach()
        x1 = x[..., 1].detach()
        expected = torch.zeros(1, 2, 2, 2)
        expected[..., 0, 0] = 2 * x0
        expected[..., 0, 1] = 0.0
        expected[..., 1, 0] = x1
        expected[..., 1, 1] = x0
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^3 -> R^3  (vector field in 3D space)                              #
    # ------------------------------------------------------------------ #

    def test_vector_3d_space_linear(self):
        """
        u = A @ x (linear map), Jacobian = A everywhere.
        Shape: x (2,4,3), u (2,4,3), J (2,4,3,3)
        """
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x = torch.randn(2, 4, 3).requires_grad_(True)
        u = x @ A.T  # (2, 4, 3)
        grad = differential.jacobian_field(x, u)  # (2, 4, 3, 3)

        assert x.shape == (2, 4, 3)
        assert u.shape == (2, 4, 3)
        assert grad.shape == (2, 4, 3, 3)

        expected = A.expand(2, 4, 3, 3)
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^2 -> R^(2x2)  (tensor field in 2D space)                         #
    # ------------------------------------------------------------------ #

    def test_tensor_field_2d_space(self):
        """
        Tensor field u: R^2 -> R^(2x2).
        u[..., i, j] = x[..., i] * x[..., j]  (outer product)
        du[..., i, j, k] = delta(j,k)*x[...,i] + delta(i,k)*x[...,j]

        Shape: x (1,2,2), u (1,2,2,2), J (1,2,2,2,2)
        """
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
        u = x.unsqueeze(-1) * x.unsqueeze(-2)  # (1, 2, 2, 2)
        grad = differential.jacobian_field(x, u)  # (1, 2, 2, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 2, 2)
        assert grad.shape == (1, 2, 2, 2, 2)

        x_d = x.detach()
        expected = torch.zeros(1, 2, 2, 2, 2)  # (N_e, N_q, i, j, k)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    expected[..., i, j, k] = (1.0 if j == k else 0.0) * x_d[..., i] + (
                        1.0 if i == k else 0.0
                    ) * x_d[..., j]
        assert grad.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # Higher-order gradients                                               #
    # ------------------------------------------------------------------ #

    def test_graph_retained_for_higher_order(self):
        """jacobian_field with create_graph=True should allow second-order gradients.
        Shape: x (1,2,1), u (1,2,1), J (1,2,1,1)
        """
        x = torch.tensor([[[1.0], [2.0]]], requires_grad=True)  # (1, 2, 1)
        u = x**3  # du/dx = 3x^2, d^2u/dx^2 = 6x
        grad = differential.jacobian_field(x, u)  # (1, 2, 1, 1)

        assert x.shape == (1, 2, 1)
        assert u.shape == (1, 2, 1)
        assert grad.shape == (1, 2, 1, 1)

        grad2 = torch.autograd.grad(grad.sum(), x, create_graph=False)[0]  # (1, 2, 1)
        assert grad2.shape == (1, 2, 1)

        expected = (6 * x).detach()
        assert grad2.detach() == pytest.approx(expected, self.relative_tolerance)
