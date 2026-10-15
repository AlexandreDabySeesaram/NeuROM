"""Gauss-Legendre quadrature of arbitrary order on the reference bar."""

import numpy as np
import torch

from neurom.reference_cells import BAR
from neurom.quadratures.quadrature_rule import QuadratureRule


class GaussLegendre1D(QuadratureRule):
    """``n``-point Gauss-Legendre rule on the reference bar (exact to degree ``2n - 1``).

    Useful as a high-order, element-independent rule for measuring interpolation/convergence
    errors (where the solve's own rule would flatter the result).

    Args:
        n_points (int): Number of Gauss points.
    """

    def __init__(self, n_points: int):
        ref = BAR

        nodes, weights = np.polynomial.legendre.leggauss(
            n_points
        )  # on [-1, 1], sum w = 2

        nodes = torch.as_tensor(nodes, dtype=torch.get_default_dtype())
        weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())
        lam1 = 0.5 * (nodes + 1.0)  # barycentric weight of vertex at +1
        points = torch.stack([1.0 - lam1, lam1], dim=1)  # (n, 2)

        super().__init__(ref, points, weights)  # weights sum = 2 = bar measure
