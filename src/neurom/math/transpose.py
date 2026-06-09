import torch

from neurom.samplings import Sampling
from neurom.apply import apply


def transpose_point(u: torch.Tensor) -> torch.Tensor:
    """Transpose a field tensor by swapping the last two dimensions.

    Args:
        u (torch.Tensor): Tensor to transpose, expected shape (d,...,d)
    Returns:
        u.transpose(-1,-2) if tensor has same dimensions for all shape entries, clone of itself, if there is one dimension.
    Raises:
        ValueError: If the tensor does not have the expected shape (d,...,d).
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

    The transpose is computed over the field dimensions for all elements and
    quadrature points N_e and N_q.

    Rules (per point):
    - Scalars (1,): unchanged
    - Vectors (d,): unchanged
    - Matrices (d, d): last two dims swapped
    - Higher-order tensors: swap last two axes

    Args:
        u (Sampling): Sampling whose field tensors will be transposed, of shape (*batch_shape, *f_shape).
    Returns:
        A Sampling with the same type and batch_shape as ``u``, with the last two field dimensions swapped where defined.
    """
    return apply(transpose_point, u)
