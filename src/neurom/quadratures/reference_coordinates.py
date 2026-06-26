"""Conversion of quadrature points to reference-element coordinates."""

import torch

from neurom.geometry.barycentric_to_reference import barycentric_to_reference
from neurom.quadratures.quadrature_rule import QuadratureRule


def reference_coordinates(n_elements: int, quad: QuadratureRule) -> torch.Tensor:
    """Compute reference-space coordinates of quadrature points for all elements.

    Converts the barycentric quadrature points stored in ``quad`` to
    reference coordinates via
    :func:`~neurom.geometry.barycentric_to_reference.barycentric_to_reference`,
    then broadcasts the result to every element.

    Args:
        n_elements (int): Number of elements; used to broadcast the output
            along the first dimension.
        quad (QuadratureRule): Quadrature rule providing barycentric points
            (shape ``(N_q, N_nodes)``) and the reference element whose
            simplex vertices are used for the conversion.

    Returns:
        torch.Tensor: Reference coordinates ``xi`` of shape
        ``(n_elements, N_q, dim_ref)``.
    """
    # (N_q, N_nodes), barycentric coordinates of the quadrature rule
    x_q_bary = quad.points()

    # (N_q, dim), reference coordinates of the reference element
    xi = barycentric_to_reference(x_lambda=x_q_bary, element=quad.reference_element)

    # Broadcast to every element: (N_e, N_q, dim)
    return xi.unsqueeze(0).expand(n_elements, -1, -1)
