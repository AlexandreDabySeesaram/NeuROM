"""Inner (contraction) product of sampled field tensors."""

import torch


def inner_point(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the inner product of two single-point field tensors.

    The inner product is the element-wise product of ``u`` and ``v`` summed
    over all field dimensions, returning a scalar wrapped in a 1-element
    tensor.

    Args:
        u (torch.Tensor): First field tensor of shape ``(*u_shape)``.
        v (torch.Tensor): Second field tensor of shape ``(*v_shape)``.
            Must satisfy ``v.shape == u.shape``.

    Returns:
        torch.Tensor: Scalar inner product of shape ``(1,)``.

    Raises:
        AssertionError: If ``u.shape != v.shape``.
    """

    assert u.shape == v.shape
    return (u * v).sum().unsqueeze(0)


def inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the inner product over all elements and quadrature points.

    The inner product is computed over the field dimensions for every
    combination of element and quadrature point.

    Args:
        u (torch.Tensor): First field tensor of shape ``(N_e, N_q, *u_shape)``.
        v (torch.Tensor): Second field tensor of shape ``(N_e, N_q, *v_shape)``.
            Must satisfy ``v.shape == u.shape``.

    Returns:
        torch.Tensor: Inner product tensor of shape ``(N_e, N_q, 1)``.

    Raises:
        ValueError: If ``u.shape != v.shape``.
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
