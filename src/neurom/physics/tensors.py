import torch

from neurom.math.jacobian import jacobian
from neurom.math.inner import inner
from neurom.math.transpose import transpose
from neurom.math.identity import identity
from neurom.math.trace import trace


def linear_elastic_stress(
    strain: torch.Tensor, lame_lambda: float, lame_mu: float
) -> torch.Tensor:
    """Consitutive law for linear elasticity"""
    trace_ = trace(strain).unsqueeze(-1)
    id_ = identity(strain.shape)
    return lame_lambda * trace_ * id_ + 2.0 * lame_mu * strain


def green_lagrange_strain(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute green lagrange strain based on displacement field"""
    du_dx = jacobian(x, u)
    epsilon = 0.5 * (du_dx + transpose(du_dx))
    return epsilon


def stress_deviator(stress):
    return stress - 1.0 / 3.0 * trace(stress).unsqueeze(-1) * identity(stress.shape)


def stress_von_mises(stress_dev):
    from neurom.math.inner import inner

    return torch.sqrt(1.5 * inner(stress_dev, stress_dev))


def cauchy_stress(
    x: torch.Tensor, u: torch.Tensor, strain, constitutive_law
) -> torch.Tensor:
    """Compute Cauchy stress based on a displacement field and constitutive law"""
    strain_ = strain(x, u)
    stress = constitutive_law(strain_)
    return stress
