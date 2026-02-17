from abc import ABC
import torch
import torch.nn as nn


class ReferenceElement(nn.Module, ABC):
    """
    Stores reference simplex geometry.

    simplex : (N_nodes, dim_ref)
    measure : scalar
    """

    def __init__(self, simplex: torch.Tensor, measure: torch.Tensor):
        super().__init__()

        if simplex.ndim != 2:
            raise ValueError("simplex must be (N_nodes, dim_ref)")

        self.register_buffer("simplex", simplex)
        self.register_buffer("measure", measure)


class Segment(ReferenceElement):
    """
    Segment [-1, 1] embedded in 1D reference space.

    simplex size: (N_nodes, dim_ref) = (2, 1)
    """

    def __init__(self):
        simplex = torch.tensor([[-1.0], [1.0]])  # (2,1)

        measure = torch.tensor(2.0)

        super().__init__(simplex, measure)


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
