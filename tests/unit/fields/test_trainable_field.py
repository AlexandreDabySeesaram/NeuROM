import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.fields import TrainableField, Field
from neurom.meshes import Mesh, Topology
from neurom.elements import P1_BAR, VectorElement
from neurom.function_space import FunctionSpace, DirichletBC

torch.set_default_dtype(torch.float32)


def _p1_space(N=4):
    cell_vertices = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, points, name="x")
    mesh = Mesh(topology, coords)
    return FunctionSpace(mesh.topology, P1_BAR)


@pytest.fixture
def field_no_constraint():
    """
    Prepare a TrainableField with:
    * name = "test"
    * P1 space over a 3-cell / 4-vertex bar mesh.
    * Values: [3., 7., 6., -5.]
    * No boundary conditions (all DOFs free).
    """
    space = _p1_space(4)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    field = TrainableField(space, init_values, name="test")

    return field


@pytest.fixture
def field_dirichlet_constraint():
    """
    Prepare a TrainableField with:
    * name = "test"
    * P1 space over a 3-cell / 4-vertex bar mesh.
    * Values: [3., 7., 6., -5.]
    * Dirichlet BCs: vertex 0 -> 100, vertex 2 -> 200 (one BC per imposed value).
    """
    space = _p1_space(4)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    field = TrainableField(
        space,
        init_values,
        bcs=[
            DirichletBC(0, [0], value=100.0),
            DirichletBC(0, [2], value=200.0),
        ],
        name="test",
    )

    return field


class TestTrainableFieldWithNoConstraint:
    """
    Test TrainableField class with NoConstraint

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_construction(self, field_no_constraint):
        """
        Test construction of a TrainableField.
        """
        assert field_no_constraint.name == "test"
        assert isinstance(field_no_constraint.values_reduced, nn.Parameter)

        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
        assert field_no_constraint.values_reduced.detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_full_values(self, field_no_constraint):
        """
        Test method TrainableField.full_values()
        """
        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
        assert field_no_constraint.full_values().detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def at_elements(self, field_no_constraint):
        """
        Test method TrainableField.at_elements()
        """
        # Tensor of shape (N_e, N_nodes, dim)
        expected_values = torch.tensor([[3.0, 7.0], [7.0, 6.0], [6.0, -5.0]]).unsqueeze(
            -1
        )
        assert field_no_constraint.at_elements() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )


class TestTrainableFieldWithDirichletConstraint:
    """
    Test TrainableField class with Dirichlet constraint

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_construction(self, field_dirichlet_constraint):
        """
        Test construction of a TrainableField.
        """
        assert field_dirichlet_constraint.name == "test"
        assert isinstance(field_dirichlet_constraint.values_reduced, nn.Parameter)

        expected_values = torch.tensor([7.0, -5.0]).unsqueeze(-1)
        assert field_dirichlet_constraint.values_reduced.detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_full_values(self, field_dirichlet_constraint):
        """
        Test method TrainableField.full_values()
        """
        expected_values = torch.tensor([100.0, 7.0, 200.0, -5.0]).unsqueeze(-1)
        assert field_dirichlet_constraint.full_values().detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def at_elements(self, field_dirichlet_constraint):
        """
        Test method TrainableField.at_elements()
        """
        expected_values = torch.tensor(
            [[100.0, 7.0], [7.0, 200.0], [200.0, -5.0]]
        ).unsqueeze(-1)
        assert field_dirichlet_constraint.at_elements() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )
