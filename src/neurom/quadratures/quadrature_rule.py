"""Base class for quadrature rules (points and weights on a reference element)."""

from abc import ABC

import torch

from neurom.reference_cells.reference_cell import ReferenceCell


class QuadratureRule(ABC):
    """Abstract base class for quadrature rules stored in barycentric coordinates.

    A plain immutable object: the points and weights are constant
    reference data, stored as a float64 master and returned on access in
    ``torch.get_default_dtype()``. Concrete subclasses build
    the barycentric points ``(N_q, N_nodes)`` and reference weights ``(N_q,)`` and hand them to
    ``super().__init__``.

    Attributes:
        reference_cell (ReferenceCell): The reference element the rule is defined on.
        points_barycentric (torch.Tensor): Barycentric quadrature points ``(N_q, N_nodes)`` in the
            active dtype.
        weights_ref (torch.Tensor): Reference integration weights ``(N_q,)`` in the active dtype.
    """

    def __init__(self, reference_cell: ReferenceCell, points, weights):
        """Store the reference element and the rule's points/weights as float64 masters.

        Args:
            reference_cell (ReferenceCell): The reference element the rule is defined on.
            points: Barycentric quadrature points ``(N_q, N_nodes)``.
            weights: Reference integration weights ``(N_q,)``.
        """
        self.reference_cell = reference_cell
        self._points_barycentric = torch.as_tensor(points, dtype=torch.float64)
        self._weights_ref = torch.as_tensor(weights, dtype=torch.float64)

    @property
    def points_barycentric(self) -> torch.Tensor:
        """Barycentric quadrature points ``(N_q, N_nodes)`` in the active default dtype."""
        return self._points_barycentric.to(torch.get_default_dtype())

    @property
    def weights_ref(self) -> torch.Tensor:
        """Reference integration weights ``(N_q,)`` in the active default dtype."""
        return self._weights_ref.to(torch.get_default_dtype())
