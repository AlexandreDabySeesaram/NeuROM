import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.field import TrainableField
from neurom.meshes import Topology
from neurom.constraints import NoConstraint

torch.set_default_dtype(torch.float32)


@pytest.fixture
def field_no_constraint():
    """
    Prepare what is needed to define a TrainableField:
    * name = "test"
    * Simple topology: 3 elements with 4 nodes.
    * Values: [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0])
    field = TrainableField(
        name="test",
        topology=topology,
        init_values=init_values,
        constraint=NoConstraint(),
    )

    return field


class TestTrainableField:
    """
    Test TrainableField class

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

        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0])
        assert field_no_constraint.values_reduced.detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_full_values(self, field_no_constraint):
        """
        Test method TrainableField.full_values()
        """
        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0])
        assert field_no_constraint.full_values().detach() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def at_elements(self, field_no_constraint):
        """
        Test method TrainableField.at_elements()
        """

        expected_values = torch.tensor([[3.0, 7.0], [7.0, 6.0], [6.0, -5.0]])
        assert field_no_constraint.at_elements() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )
