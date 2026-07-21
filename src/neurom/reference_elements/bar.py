"""Reference bar (1D line) element."""

import torch

from neurom.reference_elements.reference_element import ReferenceElement


class Bar(ReferenceElement):
    """Reference bar element on the interval :math:`[-1, 1]` in 1-D.

    The two vertices are placed at :math:`\\xi = -1` and :math:`\\xi = 1`,
    giving a simplex of shape ``(2, 1)`` and a measure (length) of 2.

    Attributes:
        simplex (torch.Tensor): Vertex coordinates ``[[-1.0], [1.0]]``,
            shape ``(2, 1)``.
        measure (torch.Tensor): Length of the reference bar, equal to
            ``2.0``.
    """

    def __init__(self):
        """Construct the reference bar element with vertices at -1 and 1."""
        simplex = torch.tensor([[-1.0], [1.0]])  # (2,1)

        measure = torch.tensor(2.0)

        super().__init__(simplex, measure)
