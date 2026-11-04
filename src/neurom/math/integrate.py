"""Integration of sampled fields over quadrature points."""

import torch


def integrate(integrand: torch.Tensor) -> torch.Tensor:
    """Integrate a field tensor over all elements and quadrature points.

    The integrand is expected to have shape ``(N_e, N_q, *field_shape)``.
    Integration is performed by contracting the first two dimensions
    ``N_e`` and ``N_q`` via ``torch.einsum``, which is equivalent to
    summing all element and quadrature contributions.

    Args:
        integrand (torch.Tensor): The batched field tensor of shape
            ``(N_e, N_q, *field_shape)`` to integrate.

    Returns:
        torch.Tensor: The integrated result of shape ``(*field_shape)``,
        obtained by summing over the element and quadrature dimensions.
    """
    return torch.einsum("eq...->", integrand)
