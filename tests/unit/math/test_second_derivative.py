import torch
import pytest

from neurom.math import second_derivative


def test_cubic():
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1).requires_grad_()
    d2u = second_derivative(x, x[..., 0] ** 3)
    assert d2u.shape == (5, 1, 1)
    assert d2u.flatten().tolist() == pytest.approx((6 * x).flatten().tolist(), rel=1e-12)


def test_linear_is_zero():
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1).requires_grad_()
    d2u = second_derivative(x, 2 * x[..., 0])
    assert torch.allclose(d2u, torch.zeros_like(d2u))
