from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from neurom.reference_elements.reference_element import ReferenceElement


class ShapeFunction(nn.Module, ABC):
    """Base shape function class

    Provide a basis for all shape functions.

    Args:
        reference_element (ReferenceElement): The reference element on which the shape function is applied.

    Attributes:
        reference_element (ReferenceElement): The reference element on which the shape function is applied.
    """

    def __init__(self, reference_element: ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    @abstractmethod
    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Shape function

        Args:
            xi (torch.Tensor) : The reference coordinate, tensor of shape (N_e, N_q, dim_ref).
        Returns:
            Shape function evaluated at `xi`, tensor of shpae (N_e, N_q, N_nodes)
        """
        pass
