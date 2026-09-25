import pytest
import torch

from neurom.math import integrate
from neurom.physics import MembraneStretchEnergy
from neurom.physics.term import Term


class FixedTerm(Term):
    """Term returning a given integrand, standing for ``w'^2 dx``."""

    def __init__(self, values):
        self.values = values

    def integrand(self, field_layout):
        return self.values


def stretch_x2(N_e=4):
    """``w'^2 dx`` for ``w = x²`` on [0, 1], two Gauss points per element."""
    h = 1.0 / N_e
    left = torch.arange(N_e, dtype=torch.float64).unsqueeze(-1) * h
    xi = torch.tensor([-1.0, 1.0], dtype=torch.float64) / 3**0.5
    x_q = left + h / 2 * (1 + xi)  # (N_e, 2)
    return (2 * x_q) ** 2 * (h / 2)  # Gauss weights are 1


def test_energy_w_x2():
    # S = int 4x^2 = 4/3, energy = S^2 / 8 = 2/9
    energy = integrate(MembraneStretchEnergy(FixedTerm(stretch_x2())).integrand(None))
    assert energy.item() == pytest.approx(2 / 9, rel=1e-12)


def test_gradient_matches_finite_differences():
    stretch = torch.rand(4, 2, dtype=torch.float64, requires_grad=True)
    energy = lambda h: integrate(MembraneStretchEnergy(FixedTerm(h)).integrand(None))
    assert torch.autograd.gradcheck(energy, (stretch,))
