from abc import ABC
import torch
import torch.nn as nn

from neurom.reference_elements.segment import Segment
from neurom.quadratures.quadrature_rule import QuadratureRule


class MidPoint1D(QuadratureRule):
    """
    1-point midpoint quadrature in barycentric coordinates.
    """

    def __init__(self):
        ref = Segment()
        super().__init__(ref)

        # midpoint barycentric
        points = torch.tensor([[0.5, 0.5]])  # (1,2)

        weights = ref.measure[None]  # (1,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)
