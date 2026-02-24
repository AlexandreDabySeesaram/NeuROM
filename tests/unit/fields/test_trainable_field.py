import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.fields import TrainableField
from neurom.meshes import Topology
from neurom.constraints import NoConstraint, Dirichlet

torch.set_default_dtype(torch.float32)


@pytest.fixture
def field_no_constraint():
    """
    Prepare a TrainableField with:
    * name = "test"
    * Simple topology: 3 elements with 4 nodes.
    * Values: [3., 7., 6., -5.]
    * Constraint: NoConstraint
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    field = TrainableField(
        name="test",
        topology=topology,
        init_values=init_values,
        constraint=NoConstraint(),
    )

    return field


@pytest.fixture
def field_dirichlet_constraint():
    """
    Prepare a TrainableField with:
    * name = "test"
    * Simple topology: 3 elements with 4 nodes.
    * Values: [3., 7., 6., -5.]
    * Constraint: Dirichlet with nodes=[0, 2], values_imposed=[100., 200.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    values_imposed = torch.tensor([100.0, 200.0]).unsqueeze(-1)
    field = TrainableField(
        name="test",
        topology=topology,
        init_values=init_values,
        constraint=Dirichlet(nodes=[0, 2], values_imposed=values_imposed),
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
