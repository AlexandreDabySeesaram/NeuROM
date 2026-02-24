from abc import ABC
import torch
import torch.nn as nn

from neurom.reference_elements.reference_element import ReferenceElement


class QuadratureRule(nn.Module, ABC):
    """
    Stores quadrature points in barycentric coordinates.

    Attributes:
        reference_element : The ReferenceElement on which the QuadratureRule is defined
        points_barycentric : (N_q, N_nodes)
        weights            : (N_q,)

    Args:
        reference_element (ReferenceElement): The ReferenceElement on which the QuadratureRule is defined
    """

    def __init__(self, reference_element: ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    def points(self):
        return self.points_barycentric

    def weights(self):
        return self.weights_ref
