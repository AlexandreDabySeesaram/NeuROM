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
