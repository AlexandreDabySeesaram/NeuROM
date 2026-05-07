import math
import torch

from neurom.reference_elements.reference_element import ReferenceElement


class Triangle(ReferenceElement):
    """
    Reference triangle with nodes: (0,0), (0,1), (1,0)

    simplex size: (N_nodes, dim_ref) = (3, 2)
    """

    def __init__(self):
        simplex = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # (3,2)

        measure = torch.tensor(0.5 * math.sqrt(3.0))

        super().__init__(simplex, measure)
