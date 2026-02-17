from abc import ABC
import torch
import torch.nn as nn

import hidenn_playground.elements as elements


class QuadratureRule(nn.Module, ABC):
    """
    Stores quadrature points in barycentric coordinates.

    points_barycentric : (N_q, N_nodes)
    weights            : (N_q,)
    """

    def __init__(self, reference_element: elements.ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    def points(self):
        return self.points_barycentric

    def weights(self):
        return self.weights_ref


class MidPoint1D(QuadratureRule):
    """
    1-point midpoint quadrature in barycentric coordinates.
    """

    def __init__(self):
        ref = elements.Segment()
        super().__init__(ref)

        # midpoint barycentric
        points = torch.tensor([[0.5, 0.5]])  # (1,2)

        weights = ref.measure[None]  # (1,)

        self.register_buffer("points_barycentric", points)
        self.register_buffer("weights_ref", weights)


class TwoPoints1D(QuadratureRule):
    """
    2-point Gauss quadrature stored in barycentric coordinates.
    """

    def __init__(self):
        ref = elements.Segment()
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
