"""Unit tests for sampling classes.

This module verifies that the sampling data structures correctly enforce tensor
shape constraints and expose expected properties such as ``shape``, ``ndim``,
``batch_shape`` and ``f_shape``.
"""

import pytest
import torch

# Import library modules
from neurom.samplings import (
    NodalSampling,
    ElementSampling,
    QuadratureSampling,
)

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


class TestSamplingOperators:
    """
    Test the arithmetic operators of Sampling.

    Operators must preserve the concrete Sampling type and operate on values.
    """

    relative_tolerance = 1e-9

    def _make(self):
        return NodalSampling(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_add(self):
        a = self._make()
        b = self._make()
        c = a + b
        assert type(c) is NodalSampling
        assert c.values == pytest.approx(2.0 * a.values, rel=self.relative_tolerance)

    def test_sub(self):
        a = self._make()
        b = self._make()
        c = a - b
        assert type(c) is NodalSampling
        assert c.values == pytest.approx(
            torch.zeros_like(a.values), abs=self.relative_tolerance
        )

    def test_neg(self):
        a = self._make()
        c = -a
        assert type(c) is NodalSampling
        assert c.values == pytest.approx(-a.values, rel=self.relative_tolerance)

    def test_mul_scalar(self):
        a = self._make()
        c = a * 2.0
        assert type(c) is NodalSampling
        assert c.values == pytest.approx(2.0 * a.values, rel=self.relative_tolerance)

    def test_rmul_scalar(self):
        a = self._make()
        c = 2.0 * a
        assert type(c) is NodalSampling
        assert c.values == pytest.approx(2.0 * a.values, rel=self.relative_tolerance)

    def test_add_non_sampling_raises(self):
        a = self._make()
        with pytest.raises(TypeError):
            _ = a + 1.0

    def test_sub_non_sampling_raises(self):
        a = self._make()
        with pytest.raises(TypeError):
            _ = a - 1.0
