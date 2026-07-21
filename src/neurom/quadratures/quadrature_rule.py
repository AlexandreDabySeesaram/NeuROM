"""Base class for quadrature rules (points and weights on a reference element)."""

from abc import ABC
import torch.nn as nn

from neurom.reference_elements.reference_element import ReferenceElement


class QuadratureRule(nn.Module, ABC):
    """Abstract base class for quadrature rules stored in barycentric coordinates.

    Concrete subclasses register two buffers:

    - ``points_barycentric`` — quadrature points as barycentric coordinates,
      shape ``(N_q, N_nodes)``.
    - ``weights_ref`` — integration weights in the reference element,
      shape ``(N_q,)``.

    The class inherits from ``torch.nn.Module`` so that the buffers are
    managed by PyTorch (device placement, serialisation, etc.).

    Attributes:
        reference_element (ReferenceElement): The reference element on which
            the quadrature rule is defined.
        points_barycentric (torch.Tensor): Barycentric coordinates of the
            quadrature points, shape ``(N_q, N_nodes)``.
        weights_ref (torch.Tensor): Integration weights in the reference
            element, shape ``(N_q,)``.
    """

    def __init__(self, reference_element: ReferenceElement):
        """Initialise the base quadrature rule.

        Args:
            reference_element (ReferenceElement): The reference element on
                which the quadrature rule is defined.
        """
        super().__init__()
        self.reference_element = reference_element

    def points(self):
        """Return the quadrature points in barycentric coordinates.

        Returns:
            torch.Tensor: Barycentric coordinates of shape
            ``(N_q, N_nodes)``.
        """
        return self.points_barycentric

    def weights(self):
        """Return the quadrature weights in the reference element.

        Returns:
            torch.Tensor: Integration weights of shape ``(N_q,)``.
        """
        return self.weights_ref
