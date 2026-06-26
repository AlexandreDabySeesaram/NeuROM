"""Transpose of sampled field tensors."""

import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def transpose_point(u: torch.Tensor) -> torch.Tensor:
    """Transpose a single-point field tensor by swapping the last two dimensions.

    For a 1-D tensor (scalar or vector) the tensor is returned unchanged
    (clone).  For a tensor whose every dimension has the same size ``d``
    (i.e. shape ``(d, ..., d)``), the last two axes are swapped via
    ``u.transpose(-1, -2)``.

    Args:
        u (torch.Tensor): Field tensor to transpose, expected to have shape
            ``(d,)`` (vector, returned as-is) or ``(d, ..., d)`` (uniform
            square tensor).

    Returns:
        torch.Tensor: Transposed tensor with the last two dimensions swapped,
        or a clone of ``u`` when ``u`` is 1-D.

    Raises:
        ValueError: If the tensor does not have shape ``(d,)`` or ``(d, ..., d)``.
    """
    # Scalar or vector case -> return itself
    if u.ndim == 1:
        return u.clone()

    # If tensor has same dimension for all entries of shape, transpose last two indices
    shape = u.shape
    if all(s == shape[0] for s in shape):
        return u.transpose(-1, -2)

    raise ValueError(
        f"Expected tensor with shape (d,...,d) but got '{u.shape}' instead"
    )


def transpose(u: Sampling) -> Sampling:
    """Transpose a field by swapping the last two field dimensions.

    Applies :func:`transpose_point` to every point in the sampling's batch
    dimensions.  The per-point rules are:

    - Scalars ``(1,)``: returned unchanged.
    - Vectors ``(d,)``: returned unchanged.
    - Matrices ``(d, d)``: last two dimensions swapped.
    - Higher-order uniform tensors ``(d, ..., d)``: last two axes swapped.

    Args:
        u (Sampling): Sampling whose field tensors will be transposed.
            The underlying tensor has shape ``(*batch_shape, *f_shape)``.

    Returns:
        Sampling: A Sampling with the same type and ``batch_shape`` as ``u``,
        with the last two field dimensions swapped for non-1-D field shapes.
    """
    return apply(transpose_point, u)
