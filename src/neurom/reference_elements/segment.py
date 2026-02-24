import torch

from neurom.reference_elements.reference_element import ReferenceElement


class Segment(ReferenceElement):
    """
    Segment [-1, 1] embedded in 1D reference space.

    simplex size: (N_nodes, dim_ref) = (2, 1)
    """

    def __init__(self):
        simplex = torch.tensor([[-1.0], [1.0]])  # (2,1)

        measure = torch.tensor(2.0)

        super().__init__(simplex, measure)
