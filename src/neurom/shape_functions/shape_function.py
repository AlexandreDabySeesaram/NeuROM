"""Abstract base class for element shape functions."""

from abc import ABC, abstractmethod
import torch

from neurom.reference_cells.reference_cell import ReferenceCell


class ShapeFunction(ABC):
    r"""Abstract base class for finite-element shape functions.

    Subclasses implement the ``N`` method for a specific element type and polynomial order.
    A ``ShapeFunction`` is a plain immutable object (not an ``nn.Module``): :math:`N(\xi)` is
    computed directly from the input :math:`\xi`, so it carries no dtype/device state and a single
    instance is shared across meshes.

    The DOF layout (which DOF is a value, a derivative, etc., and where it lives)
    is owned by the :class:`~neurom.elements.FiniteElement`, not the shape function;
    a ``ShapeFunction`` only evaluates the reference basis :math:`N(\xi)`.

    Attributes:
        reference_cell (ReferenceCell): The reference element on which
            the shape function is defined.
    """

    def __init__(self, reference_cell: ReferenceCell):
        """Initialise the shape function with its reference element.

        Args:
            reference_cell (ReferenceCell): The reference element on
                which the shape function is defined.
        """
        self.reference_cell = reference_cell

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
