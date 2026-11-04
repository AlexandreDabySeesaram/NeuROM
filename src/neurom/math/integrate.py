"""Integration of sampled fields over quadrature points."""

import torch


def integrate(integrand: torch.Tensor) -> torch.Tensor:
    """Integrate a field tensor over all elements and quadrature points.

    The integrand is expected to have shape ``(N_e, N_q, *field_shape)``.
    Integration is performed with ``torch.einsum("eq...->", ...)``, whose
    empty output subscript contracts **every** dimension -- the element and
    quadrature axes *and* the trailing field axes. Callers wanting a
    per-component result must therefore integrate each component separately.

    Args:
        integrand (torch.Tensor): The batched field tensor of shape
            ``(N_e, N_q, *field_shape)`` to integrate.

    Returns:
        torch.Tensor: The integral as a **0-d scalar tensor**, obtained by
        summing over all dimensions of ``integrand``.
    """
    return torch.einsum("eq...->", integrand)
