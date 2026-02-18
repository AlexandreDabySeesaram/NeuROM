from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from neurom.reference_elements.reference_element import ReferenceElement


class ShapeFunction(nn.Module, ABC):
    def __init__(self, reference_element: ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    @abstractmethod
    def N(self, xi_q: torch.Tensor) -> torch.Tensor:
        """
        xi_q : (N_q, dim_ref)
        returns (N_q, N_nodes)
        """
        pass
