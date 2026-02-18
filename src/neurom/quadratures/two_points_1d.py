from abc import ABC
import torch
import torch.nn as nn

from neurom.reference_elements.segment import Segment
from neurom.quadratures.quadrature_rule import QuadratureRule


class TwoPoints1D(QuadratureRule):
    """
    2-point Gauss quadrature stored in barycentric coordinates.
    """

    def __init__(self):
        ref = Segment()
        super().__init__(ref)

        # Gauss points in barycentric coordinates
        a = 0.5 * (1.0 - 1.0 / torch.sqrt(torch.tensor(3.0)))
        b = 0.5 * (1.0 + 1.0 / torch.sqrt(torch.tensor(3.0)))

        points = torch.stack(
            [
                torch.tensor([b, a]),
                torch.tensor([a, b]),
            ]
        )  # (2,2)

        weights = 0.5 * ref.measure * torch.ones(2)  # (2,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
