from abc import ABC
import torch
import torch.nn as nn

import neurom.reference_elements as reference_elements


class QuadratureRule(nn.Module, ABC):
    """
    Stores quadrature points in barycentric coordinates.

    points_barycentric : (N_q, N_nodes)
    weights            : (N_q,)
    """

    def __init__(self, reference_element: reference_elements.ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    def points(self):
        return self.points_barycentric

    def weights(self):
        return self.weights_ref
