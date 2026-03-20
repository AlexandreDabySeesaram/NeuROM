import pytest
import torch

from neurom.physics.elastic_energy import ElasticEnergy
from neurom.field_layout import FieldLayout
from neurom.interpolation.quadrature_interpolation_result import (
    QuadratureInterpolationResult,
)
from neurom.fields.field_base import FieldBase

torch.set_default_dtype(torch.float32)


class DummyField(FieldBase):
    """Minimal concrete ``FieldBase`` implementation for testing.

    Only the ``name`` attribute is required for the ``ElasticEnergy`` term.
    """

    def __init__(self, name: str):
        # ``topology`` is unused in these tests; ``None`` is acceptable.
        super().__init__(name=name, topology=None)

    def full_values(self):
        return torch.tensor([])

    def at_elements(self):
        return torch.tensor([])


class TestElasticEnergy:
    """Test suite for the :class:`neurom.physics.elastic_energy.ElasticEnergy`.

    The suite verifies that the integrand correctly computes the elastic
    energy density ``½‖∇u‖² dx`` for simple, analytically tractable cases.
    """

    #: Relative tolerance used for all numeric approximations.
    relative_tolerance = 1e-6

    def _setup_layout(
        self, field: DummyField, x: torch.Tensor, u: torch.Tensor, dx: torch.Tensor
    ) -> FieldLayout:
        """Create a ``FieldLayout`` with a single field and its interpolation.

        Args:
            field: The dummy field to register.
            x: Tensor of quadrature coordinates, shape ``(N_e, N_q, d)``.
            u: Tensor of field values at quadrature points, matching ``x``.
            dx: Quadrature measure tensor, shape ``(N_e, N_q, 1)``.

        Returns:
            A ``FieldLayout`` containing the provided interpolation result.
        """
        layout = FieldLayout()
        layout.add(field)
        result = QuadratureInterpolationResult(x=x, u=u, measure=dx)
        layout.update(field, result)
        return layout

    def test_linear_scalar_field(self):
        """Elastic energy for a non‑constant scalar field on multiple elements.

        We use a 1‑D domain split into two elements with one quadrature point
        The field is ``u(x) = x**2`` so that ``∇u = 2x`` and the elastic energy density is ``½‖∇u‖² = 2x²``.
        With a constant measure ``dx = 0.5`` per element the analytical energy is x².
        """

        field = DummyField(name="u")
        # Three elements, two quadratures point each, 1‑D coordinates
        x = torch.tensor([3.0, 2.0, 2.0, 4.0, 5.0, -6.0], requires_grad=True).reshape(
            3, 2, 1
        )  # (3,2,1)
        u = x**2  # u = x^2, retains graph
        dx = 0.5 * torch.ones(3, 2, 1)  # (3,2,1)

        assert x.shape == (3, 2, 1)
        assert u.shape == (3, 2, 1)

        layout = self._setup_layout(field, x, u, dx)
        term = ElasticEnergy(field)
        result = term.integrand(layout)

        # 0.5 * (u'(x)))**2 = 2 * x**2
        # dx = 0.5 -> expected = x**2
        expected = x**2
        assert expected.shape == (3, 2, 1)
        assert result.detach() == pytest.approx(
            expected.detach(), rel=self.relative_tolerance
        )

    def test_linear_scalar_field_one_quadrature_point(self):
        """Elastic energy for a non‑constant scalar field on multiple elements.

        Check behavior in case of a single quadrature point to check we don't accidently squeeze out dimensions.
        We use a 1‑D domain split into two elements with one quadrature point
        The field is ``u(x) = x**2`` so that ``∇u = 2x`` and the elastic energy density is ``½‖∇u‖² = 2x²``.
        With a constant measure ``dx = 0.5`` per element the analytical energy is x².
        """

        field = DummyField(name="u")
        # Three elements, two quadratures point each, 1‑D coordinates
        x = torch.tensor([3.0, 2.0, 2.0], requires_grad=True).reshape(
            3, 1, 1
        )  # (3,1,1)
        u = x**2  # u = x^2, retains graph
        dx = 0.5 * torch.ones(3, 1, 1)  # (3,1,1)

        assert x.shape == (3, 1, 1)
        assert u.shape == (3, 1, 1)

        layout = self._setup_layout(field, x, u, dx)
        term = ElasticEnergy(field)
        result = term.integrand(layout)

        # 0.5 * (u'(x)))**2 = 2 * x**2
        # dx = 0.5 -> expected = x**2
        expected = x**2
        assert expected.shape == (3, 1, 1)
        assert result.detach() == pytest.approx(
            expected.detach(), rel=self.relative_tolerance
        )

    def test_vector_field_2d(self):
        """Elastic energy for a simple 2‑D vector field.

        Let ``u = [x**2, y**2]`` on a 2‑D quadrature point with 2 elements.
        Then the elastic energy is 0.5*4(x**2+y**2).
        With a constant measure ``dx = 0.5`` per element the analytical energy is x²+y².
        """
        field = DummyField(name="u_vec")
        x = torch.tensor(
            [
                [[3.0, 2.0], [4.0, 45.0]],
                [[1.0, -4.0], [-4.0, 55.0]],
                [[7.0, 5.0], [4.0, -9.0]],
            ],
            requires_grad=True,
        )  # (3,2,2)
        # Define u(x,y) = (x**2, y**2)
        u = torch.zeros(3, 2, 2)
        u[:, :, 0] = x[:, :, 0] ** 2
        u[:, :, 1] = x[:, :, 1] ** 2

        dx = 0.5 * torch.ones(3, 2, 1)  # (3,2,1)

        assert x.shape == (3, 2, 2)
        assert u.shape == (3, 2, 2)

        layout = self._setup_layout(field, x, u, dx)

        term = ElasticEnergy(field)
        result = term.integrand(layout)

        expected = (x[:, :, 0] ** 2 + x[:, :, 1] ** 2).unsqueeze(-1)
        assert result.detach() == pytest.approx(
            expected.detach(), rel=self.relative_tolerance
        )
