import pytest
import torch

# Import library modules
from neurom.shape_functions import LinearBar, QuadraticBar, HermiteBeam
from neurom.meshes import Connectivity, Mesh
from neurom.fields.field import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.dof_transformations import (
    IdentityDofTransformation,
    NodalDerivativeDofTransformation,
    default_dof_transformation,
)

torch.set_default_dtype(torch.float32)


@pytest.fixture
def mapping():
    """
    Prepare a two-element non-uniform mesh [0, 0.25, 1] and its mapping.
    """
    nodes = torch.tensor([0, 1, 2])
    elements = torch.tensor([[0, 1], [1, 2]])
    connectivity = Connectivity(nodes, elements)
    x = Field(
        name="x", connectivity=connectivity, values=torch.tensor([[0.0], [0.25], [1.0]])
    )
    mesh = Mesh(connectivity=connectivity, nodes_positions=x)
    return IsoparametricMapping1D(LinearBar(), mesh)


@pytest.fixture
def hermite_transformation(mapping):
    """
    Nodal derivative transformation with the Hermite beam DOF description.
    """
    sf = HermiteBeam()
    return NodalDerivativeDofTransformation(mapping, sf.dof_kinds, sf.dof_nodes_xi)


class TestIdentityDofTransformation:
    """
    Test IdentityDofTransformation class
    """

    def test_returns_dofs_unchanged(self):
        """
        Nodal values are left untouched
        """
        u_elem = torch.rand(3, 2, 1)

        assert IdentityDofTransformation().to_reference(u_elem) is u_elem


class TestNodalDerivativeDofTransformation:
    """
    Test NodalDerivativeDofTransformation class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-6

    def test_matrix_on_non_uniform_mesh(self, hermite_transformation):
        """
        M_e = diag(1, h_e/2, 1, h_e/2) with h = (0.25, 0.75)
        """
        # (N_e, N_dofs, dim) = (2, 4, 1)
        u_elem = torch.ones(2, 4, 1)

        u_ref = hermite_transformation.to_reference(u_elem)

        u_ref_expected = torch.tensor(
            [[1.0, 0.125, 1.0, 0.125], [1.0, 0.375, 1.0, 0.375]]
        ).unsqueeze(-1)

        assert u_ref == pytest.approx(u_ref_expected, rel=self.relative_tolerance)

    def test_element_subset(self, hermite_transformation):
        """
        element_ids selects the Jacobian of the matching elements
        """
        u_elem = torch.ones(1, 4, 1)

        u_ref = hermite_transformation.to_reference(
            u_elem, element_ids=torch.tensor([1])
        )

        u_ref_expected = torch.tensor([[1.0, 0.375, 1.0, 0.375]]).unsqueeze(-1)

        assert u_ref == pytest.approx(u_ref_expected, rel=self.relative_tolerance)

    def test_broadcast_over_field_components(self, hermite_transformation):
        """
        Every field component of a derivative DOF is scaled the same way
        """
        u_elem = torch.ones(2, 4, 3)

        u_ref = hermite_transformation.to_reference(u_elem)

        assert u_ref.shape == (2, 4, 3)
        assert u_ref[1, 1] == pytest.approx(
            torch.full((3,), 0.375), rel=self.relative_tolerance
        )

    def test_rejects_unknown_dof_kind(self, mapping):
        """
        Only nodal values and first derivatives are supported
        """
        with pytest.raises(ValueError):
            NodalDerivativeDofTransformation(mapping, ("value", "d2"), (-1.0, 1.0))

    def test_rejects_length_mismatch(self, mapping):
        """
        One node coordinate is required per DOF
        """
        with pytest.raises(ValueError):
            NodalDerivativeDofTransformation(mapping, ("value", "d1"), (-1.0,))


class TestDefaultDofTransformation:
    """
    Test default_dof_transformation factory
    """

    @pytest.mark.parametrize("sf", [LinearBar(), QuadraticBar()])
    def test_lagrange_gives_identity(self, sf, mapping):
        """
        Lagrange elements need no transformation
        """
        assert isinstance(
            default_dof_transformation(sf, mapping), IdentityDofTransformation
        )

    def test_hermite_gives_nodal_derivative(self, mapping):
        """
        Hermite beam slope DOFs are scaled by the Jacobian
        """
        assert isinstance(
            default_dof_transformation(HermiteBeam(), mapping),
            NodalDerivativeDofTransformation,
        )
