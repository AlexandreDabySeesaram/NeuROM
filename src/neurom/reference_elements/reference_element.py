"""Abstract base class for reference elements."""

from abc import ABC
import torch
import torch.nn as nn


class ReferenceElement(nn.Module, ABC):
    """Abstract base class that stores reference simplex geometry.

    Subclasses define a specific reference element (e.g. bar, triangle) by
    supplying the vertex coordinates and the measure of the element.

    Attributes:
        simplex (torch.Tensor): Vertex coordinates of the reference simplex,
            shape ``(N_nodes, dim_ref)``.
        measure (torch.Tensor): Measure (length, area, or volume) of the
            reference element, scalar tensor.
    """

    def __init__(self, simplex: torch.Tensor, measure: torch.Tensor):
        """Initialise the reference element with simplex vertices and measure.

        Args:
            simplex (torch.Tensor): Vertex coordinates of shape
                ``(N_nodes, dim_ref)``.
            measure (torch.Tensor): Scalar tensor representing the measure
                (length, area, or volume) of the reference element.

        Raises:
            ValueError: If ``simplex`` is not a 2-D tensor.
        """
        super().__init__()

        if simplex.ndim != 2:
            raise ValueError("simplex must be (N_nodes, dim_ref)")

        self.register_buffer("simplex", simplex)
        self.register_buffer("measure", measure)
