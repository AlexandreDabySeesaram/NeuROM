import torch

from neurom.reference_elements.reference_element import ReferenceElement


def barycentric_to_reference(
    x_lambda: torch.Tensor, element: ReferenceElement
) -> torch.Tensor:
    """
    Convert barycentric coordinates to reference coordinates.

    x_lambda : (N_q, N_nodes)
    simplex  : (N_nodes, dim_ref)

    returns  : (N_q, dim_ref)
    """

    return torch.einsum("qn,nd->qd", x_lambda, element.simplex)
