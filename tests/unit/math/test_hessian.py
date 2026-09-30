import pytest
import torch

# Import library modules
from neurom.math.hessian import hessian

torch.set_default_dtype(torch.float32)


class TestHessian:
    """Test hessian computation for fields R^d -> R^m.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    # ------------------------------------------------------------------ #
    # R^1 -> R^1  (scalar field in 1D space)                              #
    # ------------------------------------------------------------------ #

    def test_scalar_1d_space_exp(self):
        """u = exp(x), d2u/dx2 = exp(x). Shape: x (3,2,1), u (3,2,1), H (3,2,1,1,1)"""
        x = torch.tensor(
            [[-3.0, 7.0], [10.0, 5.0], [6.0, -9.0]], requires_grad=True
        ).unsqueeze(-1)  # (3, 2, 1)
        u = torch.exp(x)  # (3, 2, 1)
        hess = hessian(x, u)  # (3, 2, 1, 1, 1)

        assert x.shape == (3, 2, 1)
        assert u.shape == (3, 2, 1)
        assert hess.shape == (3, 2, 1, 1, 1)
        assert hess.detach() == pytest.approx(
            u.unsqueeze(-1).unsqueeze(-1).detach(), self.relative_tolerance
        )

    def test_scalar_1d_space_polynomial(self):
        """u = x^3, d2u/dx2 = 6x. Shape: x (2,3,1), u (2,3,1), H (2,3,1,1,1)"""
        x = torch.linspace(-2.0, 2.0, 6).reshape(2, 3, 1).requires_grad_(True)
        u = x**3
        hess = hessian(x, u)

        assert x.shape == (2, 3, 1)
        assert u.shape == (2, 3, 1)
        assert hess.shape == (2, 3, 1, 1, 1)

        expected = (6 * x).unsqueeze(-1).unsqueeze(-1).detach()
        assert hess.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^2 -> R^1  (scalar field in 2D space)                              #
    # ------------------------------------------------------------------ #

    def test_scalar_2d_space_linear(self):
        """u = 2*x0 + 3*x1, H = 0. Shape: x (2,2,2), u (2,2,1), H (2,2,1,2,2)"""
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )  # (2, 2, 2)
        u = 2.0 * x[..., :1] + 3.0 * x[..., 1:]  # (2, 2, 1)
        hess = hessian(x, u)  # (2, 2, 1, 2, 2)

        assert x.shape == (2, 2, 2)
        assert u.shape == (2, 2, 1)
        assert hess.shape == (2, 2, 1, 2, 2)
        assert torch.allclose(hess, torch.zeros(2, 2, 1, 2, 2))

    def test_scalar_2d_space_nonlinear(self):
        """u = x0^2 * x1, H = [[2*x1, 2*x0], [2*x0, 0]]. Shape: x (1,2,2), u (1,2,1), H (1,2,1,2,2)"""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
        u = (x[..., :1] ** 2) * x[..., 1:]  # (1, 2, 1)
        hess = hessian(x, u)  # (1, 2, 1, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 1)
        assert hess.shape == (1, 2, 1, 2, 2)

        x0 = x[..., 0].detach()
        x1 = x[..., 1].detach()
        expected = torch.zeros(1, 2, 1, 2, 2)
        expected[..., 0, 0, 0] = 2 * x1
        expected[..., 0, 0, 1] = 2 * x0
        expected[..., 0, 1, 0] = 2 * x0
        expected[..., 0, 1, 1] = 0.0
        assert hess.detach() == pytest.approx(expected, self.relative_tolerance)

    def test_scalar_2d_space_exp_sin(self):
        """u = exp(x0)*sin(x1), H = exp(x0)*[[sin(x1), cos(x1)], [cos(x1), -sin(x1)]].
        Shape: x (1,2,2), u (1,2,1), H (1,2,1,2,2)
        """
        x = torch.tensor([[[0.5, 1.0], [1.0, 0.5]]], requires_grad=True)  # (1, 2, 2)
        u = torch.exp(x[..., :1]) * torch.sin(x[..., 1:])  # (1, 2, 1)
        hess = hessian(x, u)  # (1, 2, 1, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 1)
        assert hess.shape == (1, 2, 1, 2, 2)

        x0 = x[..., 0].detach()
        x1 = x[..., 1].detach()
        expected = torch.zeros(1, 2, 1, 2, 2)
        expected[..., 0, 0, 0] = torch.exp(x0) * torch.sin(x1)
        expected[..., 0, 0, 1] = torch.exp(x0) * torch.cos(x1)
        expected[..., 0, 1, 0] = torch.exp(x0) * torch.cos(x1)
        expected[..., 0, 1, 1] = -torch.exp(x0) * torch.sin(x1)
        assert hess.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^2 -> R^2  (vector field in 2D space)                              #
    # ------------------------------------------------------------------ #

    def test_vector_2d_space_linear(self):
        """
        u = [2*x0 + x1, x0 - 3*x1], H = 0.
        Shape: x (2,2,2), u (2,2,2), H (2,2,2,2,2)
        """
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [-1.0, 2.0]]], requires_grad=True
        )  # (2, 2, 2)
        u = torch.cat(
            [2.0 * x[..., :1] + x[..., 1:], x[..., :1] - 3.0 * x[..., 1:]], dim=-1
        )  # (2, 2, 2)
        hess = hessian(x, u)  # (2, 2, 2, 2, 2)

        assert x.shape == (2, 2, 2)
        assert u.shape == (2, 2, 2)
        assert hess.shape == (2, 2, 2, 2, 2)
        assert torch.allclose(hess, torch.zeros(2, 2, 2, 2, 2))

    def test_vector_2d_space_nonlinear(self):
        """
        u = [x0^2, x0*x1], H_0 = [[2, 0], [0, 0]], H_1 = [[0, 1], [1, 0]].
        Shape: x (1,2,2), u (1,2,2), H (1,2,2,2,2)
        """
        x = torch.tensor([[[1.0, 2.0], [3.0, 0.5]]], requires_grad=True)  # (1, 2, 2)
        u = torch.stack([x[..., 0] ** 2, x[..., 0] * x[..., 1]], dim=-1)  # (1, 2, 2)
        hess = hessian(x, u)  # (1, 2, 2, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 2)
        assert hess.shape == (1, 2, 2, 2, 2)

        expected = torch.tensor(
            [[[2.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]
        ).expand(1, 2, 2, 2, 2)
        assert hess.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # R^3 -> R^3  (vector field in 3D space)                              #
    # ------------------------------------------------------------------ #

    def test_vector_3d_space_linear(self):
        """
        u = A @ x (linear map), H = 0 everywhere.
        Shape: x (2,4,3), u (2,4,3), H (2,4,3,3,3)
        """
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x = torch.randn(2, 4, 3).requires_grad_(True)
        u = x @ A.T  # (2, 4, 3)
        hess = hessian(x, u)  # (2, 4, 3, 3, 3)

        assert x.shape == (2, 4, 3)
        assert u.shape == (2, 4, 3)
        assert hess.shape == (2, 4, 3, 3, 3)
        assert torch.allclose(hess, torch.zeros(2, 4, 3, 3, 3))

    # ------------------------------------------------------------------ #
    # R^2 -> R^(2x2)  (tensor field in 2D space)                         #
    # ------------------------------------------------------------------ #

    def test_tensor_field_2d_space(self):
        """
        Tensor field u: R^2 -> R^(2x2).
        u[..., i, j] = x[..., i] * x[..., j]  (outer product)
        H[..., i, j, k, l] = delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k)

        Shape: x (1,2,2), u (1,2,2,2), H (1,2,2,2,2,2)
        """
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
        u = x.unsqueeze(-1) * x.unsqueeze(-2)  # (1, 2, 2, 2)
        hess = hessian(x, u)  # (1, 2, 2, 2, 2, 2)

        assert x.shape == (1, 2, 2)
        assert u.shape == (1, 2, 2, 2)
        assert hess.shape == (1, 2, 2, 2, 2, 2)

        expected = torch.zeros(1, 2, 2, 2, 2, 2)  # (N_e, N_q, i, j, k, l)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        expected[..., i, j, k, l] = (1.0 if (i == k and j == l) else 0.0) + (
                            1.0 if (i == l and j == k) else 0.0
                        )
        assert hess.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # Higher-order gradients                                               #
    # ------------------------------------------------------------------ #

    def test_graph_retained_for_higher_order(self):
        """hessian should keep the graph to allow third-order gradients.
        Shape: x (1,2,1), u (1,2,1), H (1,2,1,1,1)
        """
        x = torch.tensor([[[1.0], [2.0]]], requires_grad=True)  # (1, 2, 1)
        u = x**4  # d2u/dx2 = 12x^2, d3u/dx3 = 24x
        hess = hessian(x, u)  # (1, 2, 1, 1, 1)

        assert x.shape == (1, 2, 1)
        assert u.shape == (1, 2, 1)
        assert hess.shape == (1, 2, 1, 1, 1)

        grad3 = torch.autograd.grad(hess.sum(), x, create_graph=False)[0]  # (1, 2, 1)
        assert grad3.shape == (1, 2, 1)

        expected = (24 * x).detach()
        assert grad3.detach() == pytest.approx(expected, self.relative_tolerance)

    # ------------------------------------------------------------------ #
    # Errors                                                               #
    # ------------------------------------------------------------------ #

    def test_x_without_requires_grad_raises(self):
        """x without requires_grad must raise instead of returning zeros."""
        x = torch.linspace(0.0, 1.0, 5).unsqueeze(-1)  # (5, 1)
        with pytest.raises(RuntimeError):
            hessian(x, 2 * x)

    def test_u_independent_of_x_raises(self):
        """u independent of x must raise instead of returning zeros."""
        x = torch.linspace(0.0, 1.0, 5).unsqueeze(-1).requires_grad_(True)  # (5, 1)
        with pytest.raises(RuntimeError):
            hessian(x, torch.ones(5, 1))
