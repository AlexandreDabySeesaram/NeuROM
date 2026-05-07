"""Unit tests for sampling classes.

This module verifies that the sampling data structures correctly enforce tensor
shape constraints and expose expected properties such as ``shape``, ``ndim``,
``batch_shape`` and ``f_shape``.
"""

import pytest
import torch

# Import library modules
from neurom.samplings import *

torch.set_default_dtype(torch.float32)


class TestNodalSampling:
    relative_tolerance = 1e-9

    def test_invalid_shape(self):
        """Test invalid shapes for samplings. The values tensor must have at least 2 dimensions for nodal and element samplings, and at least 3 dimensions for quadrature sampling."""
        v = torch.tensor([-3.0, 7.0])
        with pytest.raises(AssertionError):
            _ = NodalSampling(v)
        with pytest.raises(AssertionError):
            _ = ElementSampling(v)
        with pytest.raises(AssertionError):
            _ = QuadratureSampling(v)

    def test_nodal_sampling(self):
        """Test invalid shapes for samplings. The values tensor must have at least 2 dimensions for nodal and element samplings, and at least 3 dimensions for quadrature sampling."""
        # Scalar
        v = torch.tensor([-3.0, 7.0]).unsqueeze(-1)
        ns = NodalSampling(v)
        assert ns.shape == (2, 1)
        assert ns.ndim == 2
        assert ns.batch_shape == (2,)
        assert ns.f_shape == (1,)

        # Tensor (3,3)
        v = torch.tensor(
            [[-3.0, 7.0, 5.0], [4.0, 5.0, 4.0], [-6.0, -66.0, 43.0]]
        ).reshape(1, 3, 3)
        ns = NodalSampling(v)
        assert ns.shape == (1, 3, 3)
        assert ns.ndim == 3
        assert ns.batch_shape == (1,)
        assert ns.f_shape == (3, 3)

    def test_field_sampling(self):
        """Test field sampling. The values tensor has shape (n_nodes, dim), where n_nodes is the number of nodes and dim is the spatial dimension (e.g., 2 for 2D, 3 for 3D)."""
        v = torch.tensor([-3.0, 7.0]).unsqueeze(-1)
        fs = ElementSampling(v)
        assert fs.shape == (2, 1)
        assert fs.ndim == 2
        assert fs.batch_shape == (2,)
        assert fs.f_shape == (1,)

    def test_quadrature_sampling(self):
        """Test quadrature sampling. The values tensor has shape (n_elements, n_quadrature, dim), where n_elements is the number of elements, n_quadrature is the number of quadrature points per element, and dim is the spatial dimension (e.g., 2 for 2D, 3 for 3D, etc.)."""
        v = torch.tensor([-3.0, 7.0]).reshape(2, 1, 1)
        qs = QuadratureSampling(v)
        assert qs.shape == (2, 1, 1)
        assert qs.ndim == 3
        assert qs.batch_shape == (2, 1)
        assert qs.f_shape == (1,)
