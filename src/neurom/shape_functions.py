from abc import ABC, abstractmethod
import torch
import torch.nn as nn

import hidenn_playground.elements as elements


class ShapeFunction(nn.Module, ABC):
    def __init__(self, reference_element: elements.ReferenceElement):
        super().__init__()
        self.reference_element = reference_element

    @abstractmethod
    def N(self, xi_q: torch.Tensor) -> torch.Tensor:
        """
        xi_q : (N_q, dim_ref)
        returns (N_q, N_nodes)
        """
        pass


class LinearSegment(ShapeFunction):
    def __init__(self):
        super().__init__(elements.Segment())

    def N(self, xi):
        """
        xi: (N_e, N_q, dim_ref)
        returns (N_e, N_q, N_nodes)
        """
        xi0 = xi[..., 0]

        return torch.stack(
            [
                -0.5 * (xi0 - 1.0),
                0.5 * (xi0 + 1.0),
            ],
            dim=-1,
        )


class QuadraticSegment(ShapeFunction):
    def __init__(self):
        super().__init__(elements.Segment())

    def N(self, xi_q):

        xi = xi_q[..., 0]

        return torch.stack(
            [
                0.5 * xi * (xi - 1.0),
                1.0 - xi**2,
                0.5 * xi * (xi + 1.0),
            ],
            dim=-1,
        )
