import pytest
import torch

# Import library modules
from neurom.shape_functions import LinearBar
from neurom.meshes import Connectivity, Mesh
from neurom.fields.field import Field
from neurom.geometry import IsoparametricMapping1D

torch.set_default_dtype(torch.float32)


@pytest.fixture
def mapping():
    """
    Prepare a mesh with single element with positions and mapping to use.
    """
    # Create a mesh with a single element: [5., 10.]
    # (N_e, N_nodes, dim) = (1,2,1)
    nodes = torch.tensor([0, 1])
    elements = torch.tensor([0, 1]).reshape(1, 2)
    connectivity = Connectivity(nodes, elements)
    values = torch.tensor([5.0, 10.0]).reshape(2, 1)
    x = Field(name="x", connectivity=connectivity, values=values)
    mesh = Mesh(connectivity=connectivity, nodes_positions=x)

    # Mapping from/to reference/physical coordinates
    mapping = IsoparametricMapping1D(LinearBar(), mesh)

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


def _non_uniform_mapping(positions):
    """
    Build a two-element mapping on the given node positions.
    """
    nodes = torch.tensor([0, 1, 2])
    elements = torch.tensor([[0, 1], [1, 2]])
    connectivity = Connectivity(nodes, elements)
    x = Field(name="x", connectivity=connectivity, values=positions.reshape(3, 1))
    mesh = Mesh(connectivity=connectivity, nodes_positions=x)
    return IsoparametricMapping1D(LinearBar(), mesh)


class TestJacobianAt:
    """
    Test IsoparametricMapping1D.jacobian_at on a non-uniform mesh

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-6

    def test_jacobian_is_half_element_length(self):
        """
        J equals h/2 in each element, at every reference point
        """
        mapping = _non_uniform_mapping(torch.tensor([0.0, 0.25, 1.0]))

        # (N_e, N_p, dim) = (2, 2, 1): both end points of each element
        xi = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1).expand(2, 2, 1)

        J = mapping.jacobian_at(xi)

        J_expected = torch.tensor([0.125, 0.375]).reshape(2, 1, 1, 1).expand(2, 2, 1, 1)

        assert J.shape == (2, 2, 1, 1)
        assert J == pytest.approx(J_expected, rel=self.relative_tolerance)

    def test_jacobian_on_element_subset(self):
        """
        element_ids restricts the computation to the selected elements
        """
        mapping = _non_uniform_mapping(torch.tensor([0.0, 0.25, 1.0]))

        xi = torch.zeros(1, 1, 1)

        J = mapping.jacobian_at(xi, element_ids=torch.tensor([1]))

        assert J.shape == (1, 1, 1, 1)
        assert J.item() == pytest.approx(0.375, rel=self.relative_tolerance)

    def test_jacobian_is_differentiable_wrt_nodes(self):
        """
        Gradients flow from J back to the node positions
        """
        xi = torch.tensor([-1.0, 0.3, 1.0], dtype=torch.float64)
        xi = xi.reshape(1, 3, 1).expand(2, 3, 1)

        def J_of_nodes(positions):
            return _non_uniform_mapping(positions).jacobian_at(xi)

        positions = torch.tensor(
            [0.0, 0.25, 1.0], dtype=torch.float64, requires_grad=True
        )

        assert torch.autograd.gradcheck(J_of_nodes, (positions,))
