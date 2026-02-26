from dataclasses import dataclass
import torch

from neurom.geometry.barycentric_to_reference import barycentric_to_reference
from neurom.quadratures.quadrature_rule import QuadratureRule


def reference_coordinates(n_elements: int, quad: QuadratureRule) -> torch.Tensor:
    """
    Args:
        n_elements (int): The number of elements for broadcasting.
        quad (QuadratureRule) : Provides barycentric points and the reference element.
    Returns
        Reference coordinates xi of shape (n_e, N_q, dim).
    """
    # (N_q, N_nodes), barycentric coordinates of the quadrature rule
    x_q_bary = quad.points()

    # (N_q, dim), reference coordinates of the reference element
    xi = barycentric_to_reference(x_lambda=x_q_bary, element=quad.reference_element)

    # Broadcast to every element: (N_e, N_q, dim)
    return xi.unsqueeze(0).expand(n_elements, -1, -1)
