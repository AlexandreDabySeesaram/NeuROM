from abc import ABC
import torch
import torch.nn as nn

from neurom.reference_elements.triangle import Triangle
from neurom.quadratures.quadrature_rule import QuadratureRule


class MidPoint2D(QuadratureRule):
    """
    1-point midpoint quadrature in barycentric coordinates for 2d case.
    """

    def __init__(self):
        ref = Triangle()
        super().__init__(ref)

        # midpoint barycentric
        points = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])  # (1,2)

        weights = torch.tensor([0.5])  # (1,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
