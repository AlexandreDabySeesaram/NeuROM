"""Abstract base class for element shape functions."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from neurom.reference_elements.reference_element import ReferenceElement


class ShapeFunction(nn.Module, ABC):
    """Abstract base class for finite-element shape functions.

    Subclasses implement the ``N`` method for a specific element type and
    polynomial order.  The class inherits from ``torch.nn.Module`` so that
    registered buffers (e.g. quadrature data) are handled automatically by
    PyTorch.

    Attributes:
        reference_element (ReferenceElement): The reference element on which
            the shape function is defined.
    """

    def __init__(self, reference_element: ReferenceElement):
        """Initialise the shape function with its reference element.

        Args:
            reference_element (ReferenceElement): The reference element on
                which the shape function is defined.
        """
        super().__init__()
        self.reference_element = reference_element

    @abstractmethod
    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, N_nodes)``.
        """
        pass
