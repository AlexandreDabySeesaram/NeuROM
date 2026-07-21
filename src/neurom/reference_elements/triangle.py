"""Reference triangle (2D simplex) element."""

import torch

from neurom.reference_elements.reference_element import ReferenceElement


class Triangle(ReferenceElement):
    """Reference triangle with vertices at ``(0,0)``, ``(1,0)``, and ``(0,1)``.

    The node ordering matches the shape function convention of
    :class:`~neurom.shape_functions.linear_triangle.LinearTriangle`:
    ``N = [1 - xi0 - xi1, xi0, xi1]``, i.e. node 0 at the origin, node 1
    at ``(1, 0)``, and node 2 at ``(0, 1)``.  Barycentric coordinate
    :math:`\\lambda_i` is therefore associated with node :math:`i`
    consistently across quadrature rules and interpolation.

    Attributes:
        simplex (torch.Tensor): Vertex coordinates
            ``[[0,0],[1,0],[0,1]]``, shape ``(3, 2)``.
        measure (torch.Tensor): Area of the reference triangle, equal to
            ``0.5``.
    """

    def __init__(self):
        """Construct the reference triangle with the standard orientation."""
        simplex = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (3,2)

        # Area of the reference simplex
        measure = torch.tensor(0.5)

        super().__init__(simplex, measure)
