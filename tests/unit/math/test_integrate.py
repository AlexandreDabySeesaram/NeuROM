import pytest
import torch

# Import library modules
from neurom.integrate import integrate
from neurom.quadratures.two_points_1d import TwoPoints1D

torch.set_default_dtype(torch.float32)


class TestIntegrate:
    """Test Integrate method

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_scalar_integrand_ones(self):
        """Integrating ones gives N_e * N_q."""
        integrand = torch.ones(3, 4)  # (N_e=3, N_q=4)

        result = integrate(integrand)

        assert integrand.shape == (3, 4)
        assert result.shape == ()
        assert result.item() == pytest.approx(12.0, self.relative_tolerance)

    def test_scalar_integrand_known_sum(self):
        """Integrand with known values sums correctly."""
        integrand = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

        result = integrate(integrand)

        assert integrand.shape == (3, 2)
        assert result.shape == ()
        assert result.item() == pytest.approx(21.0, self.relative_tolerance)
