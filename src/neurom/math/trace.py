import torch


def trace(u: torch.Tensor) -> torch.Tensor:
    """Compute trace of a field

    The trace is computed over the field dimensions for all elements and quadrature points N_e an N_q.

    Args:
        u (torch.Tensor): Tensor of which we will compute the trace (N_e, N_q, *u_shape)
    Returns:
        A torch.Tensor of shape (N_e, N_q, 1) representing the trace of the field.:w
    Raises:
        ValueError if u.shape is not a tensor
    """
    if u.ndim < 3:
        raise ValueError(
            f"Need at least 3 dimensions but got tensor with shape: '{u.shape}'"
        )

    dim = u.shape[2:]

    if dim == (1,):
        return u

    if len(dim) == 2 and dim[0] == dim[1]:
        return torch.einsum("eqii->eq", u).unsqueeze(-1)

    raise ValueError(f"Cannot compute identity for tensor with shape: '{u.shape}'")
