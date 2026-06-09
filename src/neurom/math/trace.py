"""Trace of sampled field tensors."""

import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def trace_point(u: torch.Tensor) -> torch.Tensor:
    """Compute the trace of a single-point field tensor.

    For a 1-D tensor (vector), the tensor is returned unchanged (clone).
    For a 2-D or higher tensor, the sum of the main diagonal is returned as a
    scalar wrapped in a 1-element tensor.

    Args:
        u (torch.Tensor): Single-point field tensor of shape ``(*u_shape)``.

    Returns:
        torch.Tensor: Trace value of shape ``(1,)``, or a clone of ``u`` when
        ``u`` is 1-D.
    """
    if u.ndim == 1:
        return u.clone()

    return u.diagonal().sum().reshape(1)


def trace(u: Sampling) -> Sampling:
    """Compute the trace of a field over all elements and quadrature points.

    Applies :func:`trace_point` to every point in the sampling's batch
    dimensions.  For a scalar/vector field the values are returned unchanged;
    for a matrix or higher-order field the diagonal sum is taken.

    Args:
        u (Sampling): Sampling whose field tensors have shape
            ``(*batch_shape, *u_shape)``.

    Returns:
        Sampling: A Sampling of the same type as ``u`` with field shape
        ``(*batch_shape, 1)`` containing the per-point trace values.
    """
    return apply(trace_point, u)
