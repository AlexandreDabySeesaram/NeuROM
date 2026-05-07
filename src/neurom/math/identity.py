import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def identity_point(u: torch.Tensor) -> torch.Tensor:
    """Compute identity tensor

    The identity computed from a tensor over a single point.

    Args:
        u (torch.Tensor): Single point tensor from which we want to build identity.
    Returns:
        torch.Tensor: Identity tensor of same shape than given tensor.
    Raises:
        ValueError if ``f_shape`` is not scalar or square matrix
    """
    f_shape = u.shape

    # Scalar case
    if f_shape == (1,):
        return torch.ones(1, dtype=u.dtype, device=u.device)

    # Matrix case
    if len(f_shape) == 2 and f_shape[0] == f_shape[1]:
        d = f_shape[0]
        I = torch.eye(d, dtype=u.dtype, device=u.device)
        return I

    raise ValueError(
        f"Identity only defined for scalar (1,) or square matrix (d,d), got tensor with shape: '{f_shape}'"
    )


def identity(s: Sampling) -> Sampling:
    """Compute identity tensor

    The identity is computed based on a ``sampling`` which provides a field shape ``f_shape`` and a ``batch_shape`` over which it is expanded.

    Args:
        s (Sampling): Sampling providing field and batch shape to generate the identity.
    Returns:
        torch.Tensor: Identity tensor of shape ``batch_shape + f_shape``
    Raises:
        ValueError if ``f_shape`` is not scalar or square matrix
    """
    return apply(identity_point, s)
