import torch

from neurom.reference_elements.segment import Segment
from neurom.shape_functions.shape_function import ShapeFunction


class LinearSegment(ShapeFunction):
    """Linear shape function on a Segment"""

    def __init__(self):
        super().__init__(Segment())

    def N(self, xi):
        """Shape function

        Args:
            xi (torch.Tensor) : The reference coordinate, tensor of shape (N_e, N_q, dim_ref).
        Returns:
            Shape function evaluated at `xi`, tensor of shpae (N_e, N_q, N_nodes)
        """
        xi0 = xi[..., 0]

        return torch.stack(
            [
                -0.5 * (xi0 - 1.0),
                0.5 * (xi0 + 1.0),
            ],
            dim=-1,
        )
