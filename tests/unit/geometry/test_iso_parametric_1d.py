import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D

torch.set_default_dtype(torch.float32)


@pytest.fixture
def mapping():
    """
    Prepare element with positions and mapping to use.
    """
    # Create an element: [5., 10.]
    # (N_e, N_nodes, dim) = (1,2,1)
    x_nodes = torch.tensor([[5.0, 10.0]]).reshape(1, 2, 1)

    # Shape function
    sf = LinearSegment()

    # Mapping from/to reference/physical coordinates
    mapping = IsoparametricMapping1D(sf, x_nodes)

    return mapping


class TestIsoparametricMapping1D:
    """
    Test IsoparametricMapping1D class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_map_reference_to_physical(self, mapping):
        """
        Test mapping from the reference coordinate to the physcal positions
        """
        # Check a few references coordinates: -1, -0.5, 0, 0.5, 1
        # (N_e, N_q, dim) = (1, 5, 1)
        xi = torch.tensor([[-1, -0.5, 0, 0.5, 1]]).unsqueeze(-1)

        # Compute mapping
        x = mapping.map(xi)

        # Expected positions in physical space
        x_expected = torch.tensor([[5, 6.25, 7.5, 8.75, 10.0]]).unsqueeze(-1)

        # Check values
        assert x == pytest.approx(x_expected, rel=self.relative_tolerance)

    def test_inverse_map_physical_to_reference(self, mapping):
        """
        Test mapping from the physcal positions to the reference coordinates
        """
        # Check a few physical positions: -1, -0.5, 0, 0.5, 1
        x = torch.tensor([[5, 6.25, 7.5, 8.75, 10.0]]).unsqueeze(-1)

        # Compute mapping
        xi = mapping.inverse_map(x)

        # Expected reference coordinates
        # (N_e, N_q, dim) = (1, 5, 1)
        xi_expected = torch.tensor([[-1, -0.5, 0, 0.5, 1]]).unsqueeze(-1)

        # Check values
        assert xi == pytest.approx(xi_expected, rel=self.relative_tolerance)

    def test_det_jacobian(self, mapping):
        """
        Test computation of determinant of jacobian
        """

        # Compute mapping
        det_J = mapping.det_jacobian

        det_J_expected = torch.tensor([2.5]).unsqueeze(-1)

        # Check values
        assert det_J == pytest.approx(det_J_expected, rel=self.relative_tolerance)
