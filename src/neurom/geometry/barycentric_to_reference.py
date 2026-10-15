"""Conversion from barycentric coordinates to reference-element coordinates."""

import torch

from neurom.reference_cells.reference_cell import ReferenceCell


def barycentric_to_reference(
    x_lambda: torch.Tensor, element: ReferenceCell
) -> torch.Tensor:
    """Convert barycentric coordinates to reference coordinates.

    Computes the reference-space position as a convex combination of the
    simplex vertices weighted by the barycentric coordinates, via the
    contraction :math:`\\xi_q = \\sum_n \\lambda_{qn} \\, v_n`.

    Args:
        x_lambda (torch.Tensor): Barycentric coordinates of shape
            ``(N_q, N_nodes)``, where ``N_q`` is the number of quadrature
            points and ``N_nodes`` is the number of simplex vertices.
        element (ReferenceCell): Reference element whose ``simplex``
            attribute has shape ``(N_nodes, dim_ref)``.

    Returns:
        torch.Tensor: Reference coordinates of shape ``(N_q, dim_ref)``.
    """

    # Reference vertices on demand in the inputs' dtype/device (the element pins neither).
    simplex = element.simplex.to(dtype=x_lambda.dtype, device=x_lambda.device)
    return torch.einsum("qn,nd->qd", x_lambda, simplex)
