import torch

from neurom.reference_elements.reference_element import ReferenceElement


class Triangle(ReferenceElement):
    """
    Reference triangle with nodes: (0,0), (1,0), (0,1)

    The node ordering matches the shape function convention of
    :class:`~neurom.shape_functions.linear_triangle.LinearTriangle`:
    N = [1 - xi0 - xi1, xi0, xi1], i.e. node 1 at the origin, node 2 at (1,0)
    and node 3 at (0,1). Barycentric coordinate lambda_i is therefore
    associated with node i consistently across quadrature rules and
    interpolation.

    simplex size: (N_nodes, dim_ref) = (3, 2)
    """

    def __init__(self):
        simplex = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (3,2)

        # Area of the reference simplex
        measure = torch.tensor(0.5)

        super().__init__(simplex, measure)
