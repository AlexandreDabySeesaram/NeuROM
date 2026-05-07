from abc import ABC
import torch
import torch.nn as nn

from neurom.reference_elements.triangle import Triangle
from neurom.quadratures.quadrature_rule import QuadratureRule


class ThreePoints2D(QuadratureRule):
    """
    3-point midpoint quadrature in barycentric coordinates for 2d case.
    """

    def __init__(self):
        ref = Triangle()
        super().__init__(ref)

        # three points barycentrics barycentric
        points = torch.tensor(
            [
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            ]
        )  # (3,3)

        weights = torch.tensor([1 / 6, 1 / 6, 1 / 6])

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
