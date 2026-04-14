import torch
import torch.nn as nn


def integrate(integrand: torch.Tensor):
    """Integration method

    Integrate a tensor over the elements and quadrature points. The integrand is expected to have shape (N_e, N_q, *field_shape). The integration is performed by summing over the first two dimensions (N_e and N_q).

    Args:
        integrand (torch.Tensor): The field to integrate (N_e, N_q).

    Returns:
        Result of integration computed by summing over the elements and the quadrature points.
    """
    return torch.einsum("eq...->", integrand)
