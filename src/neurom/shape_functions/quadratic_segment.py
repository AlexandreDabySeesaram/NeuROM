import torch

from neurom.reference_elements.segment import Segment
from neurom.shape_functions.shape_function import ShapeFunction


class QuadraticSegment(ShapeFunction):
    def __init__(self):
        super().__init__(Segment())

    def N(self, xi_q):

        xi = xi_q[..., 0]

        return torch.stack(
            [
                0.5 * xi * (xi - 1.0),
                1.0 - xi**2,
                0.5 * xi * (xi + 1.0),
            ],
            dim=-1,
        )
