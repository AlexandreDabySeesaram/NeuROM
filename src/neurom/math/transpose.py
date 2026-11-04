import torch


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


def transpose(u: torch.Tensor) -> torch.Tensor:
    """Transpose a field tensor by swapping the last two dimensions.

    Rules:
    - Scalars (N_e, N_q, 1): unchanged
    - Vectors (N_e, N_q, d): unchanged
    - Matrices (N_e, N_q, d, d): last two dims swapped
    - Higher-order tensors: swap last two axes

    Args:
        u (torch.Tensor): Tensor to transpose, expected shape (N_e, N_q, *u_dim)
    Returns:
        A torch.Tensor with the same shape as u, but with the last two dimensions swapped if u is a matrix or higher-order tensor.
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
