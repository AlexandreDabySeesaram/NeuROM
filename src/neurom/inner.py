import torch


def inner(u: torch.Tensor, v: torch.Tensor):
    """Compute inner product

    The inner product is computed over the field dimensions for all elements and quadrature points N_e an N_q.

    Args:
        u (torch.Tensor): First (N_e, N_q, *u_shape)
        v: (N_e, N_q, *u_shape)
    Returns:
        A torch.Tensor of shape (N_e, N_q, 1) representing the inner product over the fields dimensions.
    Raises:
        ValueError if u.shape and v.shape don't match.
    """

    u_shape = u.shape
    v_shape = v.shape

    # Same shapes?
    if u_shape != v.shape:
        raise ValueError(
            f"Trying to compute inner product with incompatible tensors, `u` has shape '{u_shape}' and `v` has shape '{v_shape}'"
        )

    # Get individual dimensions
    n_e = u_shape[0]
    n_q = u_shape[1]

    # Reshape result to tensor of shape (n_e,n_q,1)
    result = torch.einsum("eq...,eq...->eq", u, v).squeeze().reshape(n_e, n_q, 1)
    return result
