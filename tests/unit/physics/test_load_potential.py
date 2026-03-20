import pytest
import torch

from neurom.physics.load_potential import LoadPotential
from neurom.field_layout import FieldLayout
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)
from neurom.fields.field_base import FieldBase

torch.set_default_dtype(torch.float32)


class DummyField(FieldBase):
    """Minimal concrete ``FieldBase`` implementation for testing.

    Only the ``name`` attribute is required for the ``LoadPotential`` term.
    """

    def __init__(self, name: str):
        # ``topology`` is unused in these tests; ``None`` is acceptable.
        super().__init__(name=name, topology=None)

    def full_values(self):
        return torch.tensor([])

    def at_elements(self):
        return torch.tensor([])


class TestLoadPotential:
    """Test suite for :class:`neurom.physics.load_potential.LoadPotential`."""

    #: Relative tolerance for numeric comparisons.
    rel_tol = 1e-6

    def _add_to_layout(
        self,
        layout,
        field: DummyField,
        x: torch.Tensor,
        u: torch.Tensor,
        dx: torch.Tensor,
    ) -> None:
        """Create a ``FieldLayout`` containing a single ``QuadratureInterpolationResult``.

        Args:
            layout: The FieldLayout to which we will update the given field
            field: The dummy field to register.
            x: Quadrature coordinates, shape ``(N_e, N_q, d)``.
            u: Field values at quadrature points, matching ``x``.
            dx: Quadrature measure, shape ``(N_e, N_q, 1)``.
        """
        layout.add(field)
        result = QuadratureInterpolationResult(x=x, u=u, measure=dx)
        layout.update(field, result)

    def test_scalar_load(self):
        """Scalar field with a linear load density.

        ``f(x) = 2 * x`` and ``u = x``. With ``dx = 0.5`` the integrand should be
        ``-(2*x * x) * 0.5 = -x**2``.
        """
        field = DummyField(name="u")
        f_field = DummyField(name="f")
        # Three elements, two quadrature points
        x = torch.tensor([3.0, 2.0, 2.0, 4.0, 5.0, -6.0]).reshape(3, 2, 1)  # (3,2,1)
        u = x.clone()  # u = x
        f = 2.0 * x  # f = 2*x
        dx = 0.5 * torch.ones(3, 2, 1)  # (3,2,1)

        assert x.shape == (3, 2, 1)
        assert u.shape == (3, 2, 1)
        assert f.shape == (3, 2, 1)

        # Prepare layout
        layout = FieldLayout()
        self._add_to_layout(layout, field, x, u, dx)
        self._add_to_layout(layout, f_field, x, f, dx)

        term = LoadPotential(field, f_field)
        result = term.integrand(layout)

        # dx = 0.5 uniformly so expected = -0.5 * 2 * x**2 = -x**2
        expected = -(x**2)
        assert expected.shape == (3, 2, 1)
        assert result == pytest.approx(expected, rel=self.rel_tol)

    def test_scalar_load_one_quadrature_point(self):
        """Scalar field with a linear load density.

        Check behavior for one quadrature point that the final integrand we get has proper dimension (n_e,1,1) and nothing gets squeezed accidentaly.

        ``f(x) = 2 * x`` and ``u = x``. With ``dx = 0.5`` the integrand should be
        ``-(2*x * x) * 0.5 = -x**2``.
        """
        field = DummyField(name="u")
        f_field = DummyField(name="f")
        # Three elements, one quadrature point
        x = torch.tensor([3.0, 2.0, -4.0]).reshape(3, 1, 1)  # (3,1,1)
        u = x.clone()  # u = x
        f = 2.0 * x  # f = 2*x
        dx = 0.5 * torch.ones(3, 1, 1)  # (3,1,1)

        assert x.shape == (3, 1, 1)
        assert u.shape == (3, 1, 1)
        assert f.shape == (3, 1, 1)

        # Prepare layout
        layout = FieldLayout()
        self._add_to_layout(layout, field, x, u, dx)
        self._add_to_layout(layout, f_field, x, f, dx)

        term = LoadPotential(field, f_field)
        result = term.integrand(layout)

        # dx = 0.5 uniformly so expected = -0.5 * 2 * x**2 = -x**2
        expected = -(x**2)
        assert expected.shape == (3, 1, 1)
        assert result == pytest.approx(expected, rel=self.rel_tol)

    def test_vector_load(self):
        """2‑D vector field with component‑wise load.

        ``u = [x, y]`` and ``f(x, y) = [x, y]``. With ``dx = 0.2`` the integrand is
        ``- (x*x + y*y) * dx``.
        """
        field = DummyField(name="u_vec")
        f_field = DummyField(name="f_vec")
        # Two elements, one quadrature point each, 2‑D coordinates
        x = torch.tensor(
            [
                [[3.0, 2.0], [4.0, 45.0]],
                [[1.0, -4.0], [-4.0, 55.0]],
                [[7.0, 5.0], [4.0, -9.0]],
            ]
        )  # (3,2,2)
        # u = [x, y]
        u = x.clone()
        f = x.clone()  # f = x
        dx = 0.2 * torch.ones(3, 2, 1)  # (3,2,1)

        assert x.shape == (3, 2, 2)
        assert u.shape == (3, 2, 2)
        assert f.shape == (3, 2, 2)

        # Prepare layout
        layout = FieldLayout()
        self._add_to_layout(layout, field, x, u, dx)
        self._add_to_layout(layout, f_field, x, f, dx)

        term = LoadPotential(field, f_field)
        result = term.integrand(layout)

        # Expected per element: -(x**2 + y**2) * 0.2
        expected = -0.2 * (x[:, :, 0] ** 2 + x[:, :, 1] ** 2).unsqueeze(-1)
        assert expected.shape == (3, 2, 1)
        assert result == pytest.approx(expected, rel=self.rel_tol)
