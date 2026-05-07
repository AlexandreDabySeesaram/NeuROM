import torch

from neurom.math.jacobian import jacobian
from neurom.math.inner import inner, inner_point
from neurom.math.transpose import transpose, transpose_point
from neurom.math.identity import identity, identity_point
from neurom.math.trace import trace, trace_point

from neurom.samplings import Sampling
from neurom.apply import apply


def linear_elastic_stress_point(
    strain: torch.Tensor,
    lame_lambda: float,
    lame_mu: float,
) -> torch.Tensor:
    """Linear elastic stress point-wise computation

    Args:
        strain (torch.Tensor): Strain tensor of shape (d, d)
        lame_lambda (float): Lame parameter lambda
        lame_mu (float): Lame parameter mu

    Returns:
        torch.Tensor: Stress tensor of shape (d, d)
    """
    return (
        lame_lambda * trace_point(strain) * identity_point(strain)
        + 2.0 * lame_mu * strain
    )


def linear_elastic_stress(
    strain: Sampling, lame_lambda: float, lame_mu: float
) -> Sampling:
    """Consitutive law for linear elasticity

    Args:
        strain (Sampling): Strain sampling
        lame_lambda (float): Lame parameter lambda
        lame_mu (float): Lame parameter mu

    Returns:
        Sampling: Stress sampling
    """

    return apply(
        linear_elastic_stress_point,
        strain,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
    )


def green_lagrange_strain(x: Sampling, u: Sampling) -> Sampling:
    """Compute green lagrange strain based on displacement field"""
    assert type(x) == type(u), (
        f"x and u must be of the same Sampling type but got x of type '{type(x)}' and u of type '{type(u)}'"
    )

    grad = jacobian(x, u)
    return apply(lambda du_dx: 0.5 * (du_dx + transpose_point(du_dx)), grad)


def stress_deviator_point(stress: torch.Tensor) -> torch.Tensor:
    """Compute the deviatoric part of the stress tensor

    This is the single point implementation

    Args:
        stress (torch.Tensor): Cauchy stress tensor of shape (N_e, N_q, d, d)
    Returns:
        torch.Tensor: Deviatoric stress tensor of shape (N_e, N_q, d, d)
    """

    return stress - 1.0 / 3.0 * trace_point(stress) * identity_point(stress)


def stress_deviator(stress: Sampling) -> torch.Tensor:
    """Compute the deviatoric part of the stress tensor

    Args:
        stress (torch.Tensor): Cauchy stress tensor of shape (N_e, N_q, d, d)
    Returns:
        torch.Tensor: Deviatoric stress tensor of shape (N_e, N_q, d, d)
    """
    return apply(stress_deviator_point, stress)


def stress_von_mises_point(stress_dev):
    """Compute the von Mises equivalent stress from the deviatoric stress tensor

    Args:
        stress_dev (torch.Tensor): Deviatoric stress tensor of shape (N_e, N_q, d, d)
    Returns:
        torch.Tensor: Von Mises equivalent stress of shape (N_e, N_q)
    """
    return torch.sqrt(1.5 * inner_point(stress_dev, stress_dev))


def stress_von_mises(stress_dev):
    """Compute the von Mises equivalent stress from the deviatoric stress tensor

    Args:
        stress_dev (torch.Tensor): Deviatoric stress tensor of shape (N_e, N_q, d, d)
    Returns:
        torch.Tensor: Von Mises equivalent stress of shape (N_e, N_q)
    """
    return apply(stress_von_mises_point, stress_dev)
