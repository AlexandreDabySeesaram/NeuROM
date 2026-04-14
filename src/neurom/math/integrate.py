import torch
import torch.nn as nn


def integrate(integrand: torch.Tensor):
    """Integration method

    Integrate a tensor based
    Args:
        integrand (torch.Tensor): The field to integrate (N_e, N_q).
    Returns:
        Result of integration computed by summing over the elements and the quadratur epoints.
    """
    return torch.einsum("eq...->", integrand)
