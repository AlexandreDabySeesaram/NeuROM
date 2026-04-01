import torch
from typing import Callable

from neurom.differential import jacobian_field


def trace(u: torch.Tensor) -> torch.Tensor:
    """Compute trace of a field

    The inner product is computed over the field dimensions for all elements and quadrature points N_e an N_q.

    Args:
        u (torch.Tensor): First (N_e, N_q, *u_shape)
    Returns:
        A torch.Tensor of shape (N_e, N_q, 1) representing the inner product over the fields dimensions.
    Raises:
        ValueError if u.shape is not a tensor
    """
    if u.ndim < 3:
        raise ValueError(
            f"Need at least 3 dimensions but got tensor with shape: '{u.shape}'"
        )

    n_e, n_q = u.shape[:2]
    dim = u.shape[2:]

    if dim == (1,):
        return u

    if len(dim) == 2 and dim[0] == dim[1]:
        return torch.einsum("eqii->eq", u).unsqueeze(-1)

    raise ValueError(f"Cannot compute identity for tensor with shape: '{u.shape}'")


def identity(shape, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Compute identity tensor
    Assumes shape: (N_e, N_q, d, d)
    """
    if len(shape) < 3:
        raise ValueError(f"Need at least 3 dimensions but got '{shape}'")

    n_e, n_q = shape[:2]
    dim = shape[2:]

    # Scalar case
    if dim == (1,):
        return torch.ones(n_e, n_q, 1, dtype=dtype, device=device)

    # Matrix field
    if len(dim) == 2 and dim[0] == dim[1]:
        d = dim[0]
        I = torch.eye(d, dtype=dtype, device=device)
        return I.expand(n_e, n_q, d, d)

    raise ValueError(
        f"Identity only defined for scalar (1,) or square matrix (d,d), got shape: '{shape}'"
    )


def transpose(u: torch.Tensor) -> torch.Tensor:
    """
    Transpose tensorial part of a field u of shape (N_e, N_q, *u_dim).

    Rules:
    - Scalars (N_e, N_q, 1): unchanged
    - Vectors (N_e, N_q, d): unchanged
    - Matrices (N_e, N_q, d, d): last two dims swapped
    - Higher-order tensors: swap last two axes
    """

    if u.ndim < 3:
        raise ValueError(f"Expected (N_e, N_q, *u_dim), got '{u.shape}'")

    dim = u.shape[2:]

    # scalar case
    if dim == (1,):
        return u.clone()

    # vector case → no meaningful transpose
    if len(dim) == 1:
        return u.clone()

    # matrix case → standard transpose
    if len(dim) == 2:
        return u.transpose(-1, -2)

    # higher-order tensor → swap last two indices
    return u.transpose(-1, -2)


def linear_elastic_stress(
    strain: torch.Tensor, lame_lambda: float, lame_mu: float
) -> torch.Tensor:
    """Consitutive law for linear elasticity"""
    trace_ = trace(strain).unsqueeze(-1)
    id_ = identity(strain.shape)
    return lame_lambda * trace_ * id_ + 2.0 * lame_mu * strain


def green_lagrange_strain(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute green lagrange strain based on displacement field"""
    du_dx = jacobian_field(x, u)
    epsilon = 0.5 * (du_dx + transpose(du_dx))
    return epsilon


def stress_deviator(stress):
    return stress - 1.0 / 3.0 * trace(stress).unsqueeze(-1) * identity(stress.shape)


def stress_von_mises(stress_dev):
    from neurom.inner import inner

    torch.sqrt(1.5 * inner(stress_dev, stress_dev))


def cauchy_stress(
    x: torch.Tensor, u: torch.Tensor, strain: torch.Tensor, constitutive_law
) -> torch.Tensor:
    """Compute Cauchy stress based on a displacement field and constitutive law"""
    strain_ = strain(x, u)
    stress = constitutive_law(strain_)
    return stress
