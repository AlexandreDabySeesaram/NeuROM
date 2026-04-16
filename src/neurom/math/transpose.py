import torch


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
