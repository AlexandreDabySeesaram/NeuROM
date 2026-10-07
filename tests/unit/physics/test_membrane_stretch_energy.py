from types import SimpleNamespace

import pytest
import torch

from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult
from neurom.math import integrate
from neurom.physics import MembraneStretchEnergy
from neurom.samplings import QuadratureSampling


def layout(q=None, N_e=4):
    """Layout holding ``w = q x²`` on [0, 1] at two Gauss points per element (``q = 1`` by default)."""
    h = 1.0 / N_e
    left = torch.arange(N_e, dtype=torch.float64).unsqueeze(-1) * h
    xi = torch.tensor([-1.0, 1.0], dtype=torch.float64) / 3**0.5
    x = (left + h / 2 * (1 + xi)).unsqueeze(-1).requires_grad_()  # (N_e, 2, 1)
    w = x**2 if q is None else q * x**2
    measure = torch.full((N_e, 2, 1), h / 2, dtype=torch.float64)  # Gauss weights are 1
    result = QuadratureAssemblyResult(
        x=QuadratureSampling(x), u=QuadratureSampling(w), measure=QuadratureSampling(measure)
    )
    return {"w": result}


field = SimpleNamespace(name="w")


def test_energy_w_x2():
    # S = int 4x^2 = 4/3, energy = S^2 / 8 = 2/9
    energy = integrate(MembraneStretchEnergy(field, coefficient=1 / 8).integrand(layout()))
    assert energy.item() == pytest.approx(2 / 9, rel=1e-12)


def test_gradient_matches_finite_differences():
    q = torch.rand(4, 2, 1, dtype=torch.float64, requires_grad=True)
    energy = lambda q: integrate(MembraneStretchEnergy(field, coefficient=1 / 8).integrand(layout(q)))
    assert torch.autograd.gradcheck(energy, (q,))
