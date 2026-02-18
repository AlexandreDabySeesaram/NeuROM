import torch

from neurom.reference_elements.segment import Segment
from neurom.shape_functions.shape_function import ShapeFunction


class LinearSegment(ShapeFunction):
    def __init__(self):
        super().__init__(Segment())

    def N(self, xi):
        """
        xi: (N_e, N_q, dim_ref)
        returns (N_e, N_q, N_nodes)
        """
        xi0 = xi[..., 0]

        return torch.stack(
            [
                -0.5 * (xi0 - 1.0),
                0.5 * (xi0 + 1.0),
            ],
            dim=-1,
        )
