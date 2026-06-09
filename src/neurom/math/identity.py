"""Identity tensor construction over sampled fields."""

import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def identity_point(u: torch.Tensor) -> torch.Tensor:
    """Compute the identity tensor for a single point.

    Returns an identity tensor whose shape matches the field shape of ``u``.
    For a scalar input of shape ``(1,)`` this is ``torch.ones(1)``.
    For a square-matrix input of shape ``(d, d)`` this is ``torch.eye(d)``.

    Args:
        u (torch.Tensor): Single-point field tensor of shape ``(1,)`` (scalar)
            or ``(d, d)`` (square matrix).

    Returns:
        torch.Tensor: Identity tensor with the same shape as ``u``.

    Raises:
        ValueError: If ``u`` is not a scalar ``(1,)`` or a square matrix ``(d, d)``.
    """
    f_shape = u.shape

    # Scalar case
    if f_shape == (1,):
        return torch.ones(1, dtype=u.dtype, device=u.device)

    # Matrix case
    if len(f_shape) == 2 and f_shape[0] == f_shape[1]:
        d = f_shape[0]
        return torch.eye(d, dtype=u.dtype, device=u.device)

    raise ValueError(
        f"Identity only defined for scalar (1,) or square matrix (d,d), got tensor with shape: '{f_shape}'"
    )


def identity(s: Sampling) -> Sampling:
    """Compute the identity tensor for a full sampling.

    Applies :func:`identity_point` to every point in the sampling's batch
    dimensions.  The field shape ``f_shape`` must be ``(1,)`` (scalar) or
    ``(d, d)`` (square matrix).

    Args:
        s (Sampling): Sampling whose ``f_shape`` and ``batch_shape`` determine
            the output identity tensor.

    Returns:
        Sampling: A Sampling of the same type as ``s`` with the identity tensor
        broadcast over ``batch_shape``.

    Raises:
        ValueError: If ``f_shape`` is not ``(1,)`` or ``(d, d)``.
    """
    return apply(identity_point, s)
