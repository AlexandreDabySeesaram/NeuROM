import torch

from neurom.reference_elements.triangle import Triangle
from neurom.shape_functions.shape_function import ShapeFunction


class LinearTriangle(ShapeFunction):
    """Linear shape function on a Triangle"""

    def __init__(self):
        super().__init__(Triangle())

    def N(self, xi):
        """Shape function

        Args:
            xi (torch.Tensor) : The reference coordinate, tensor of shape (N_e, N_q, dim_ref).
        Returns:
            Shape function evaluated at `xi`, tensor of shape (N_e, N_q, N_nodes)
        """
        xi0 = xi[..., 0]
        xi1 = xi[..., 1]

        return torch.stack(
            [1.0 - xi0 - xi1, xi0, xi1],
            dim=-1,
        )
