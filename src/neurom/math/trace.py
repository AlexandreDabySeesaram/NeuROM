import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def trace_point(u: torch.Tensor) -> torch.Tensor:
    """Compute trace of a field

    The trace is computed over the field dimensions for a single point.

    Args:
        u (torch.Tensor): Tensor of which we will compute the trace (*u_shape)
    Returns:
        A torch.Tensor of shape (1,) representing the trace of the field.
    """
    if u.ndim == 1:
        return u.clone()

    return u.diagonal().sum()


def trace(u: Sampling) -> Sampling:
    """Compute trace of a field

    The trace is computed over the field dimensions for all elements and quadrature points N_e an N_q.

    Args:
        u (torch.Tensor): Tensor of which we will compute the trace (N_e, N_q, *u_shape)
    Returns:
        A torch.Tensor of shape (N_e, N_q, 1) representing the trace of the field.:w
    Raises:
        ValueError if u.shape is not a tensor
    """
    return apply(trace_point, u)
