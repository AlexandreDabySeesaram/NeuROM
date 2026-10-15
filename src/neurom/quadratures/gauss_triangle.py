"""High-order triangle quadrature via the Duffy (collapsed tensor-product) transform."""

import numpy as np
import torch

from neurom.reference_cells import TRIANGLE
from neurom.quadratures.quadrature_rule import QuadratureRule


class GaussTriangle(QuadratureRule):
    """Collapsed ``n_per_dir``-point Gauss rule on the reference triangle.

    A tensor Gauss rule on the unit square mapped to the triangle by the Duffy transform
    :math:`\\xi = u(1 - v),\\ \\eta = v` (Jacobian :math:`1 - v`). Integrates polynomials of
    total degree up to about :math:`2\\,\\text{n\\_per\\_dir} - 1` exactly -- a high-order,
    element-independent rule for measuring interpolation/convergence errors.

    Args:
        n_per_dir (int): Gauss points per collapsed direction (``n_per_dir**2`` total).
    """

    def __init__(self, n_per_dir: int):
        ref = TRIANGLE
        nodes, w = np.polynomial.legendre.leggauss(n_per_dir)  # [-1, 1], sum w = 2
        s = 0.5 * (nodes + 1.0)  # -> [0, 1]
        w01 = 0.5 * w  # weights on [0, 1], sum = 1
        u, v = np.meshgrid(s, s, indexing="ij")
        wu, wv = np.meshgrid(w01, w01, indexing="ij")
        u, v = u.ravel(), v.ravel()
        xi = u * (1.0 - v)  # Duffy map onto the reference triangle
        eta = v
        weights = (wu.ravel() * wv.ravel()) * (1.0 - v)  # * Duffy Jacobian; sum = 0.5

        dtype = torch.get_default_dtype()
        lam1 = torch.as_tensor(xi, dtype=dtype)
        lam2 = torch.as_tensor(eta, dtype=dtype)
        lam0 = 1.0 - lam1 - lam2
        points = torch.stack([lam0, lam1, lam2], dim=1)  # (n^2, 3) barycentric

        # weights sum = 0.5 = triangle measure
        super().__init__(ref, points, torch.as_tensor(weights, dtype=dtype))
