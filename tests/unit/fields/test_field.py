import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.fields import Field
from neurom.meshes import Topology

torch.set_default_dtype(torch.float32)


@pytest.fixture
def field():
    """
    Prepare what is needed to define a Field:
    * name = "test"
    * Simple topology: 3 elements with 4 nodes.
    * Values: [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    field = Field(name="test", topology=topology, values=values)

    return field


class TestField:
    """
    Test Field class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_construction(self, field):
        """
        Test construction of a Field.
        """
        assert field.name == "test"

        assert "values" in field._buffers
        assert isinstance(getattr(field, "values"), torch.Tensor)
        assert field.values.shape == (4, 1)
        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)

        assert field.values == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_invalid_values_shape(self, field):
        """
        Test creating a Field with invalid shape.
        """
        topology = field.topology
        values = torch.tensor([3.0, 7.0, 6.0, -5.0])
        with pytest.raises(ValueError):
            wrong_field = Field(
                name="missing field dimension", topology=topology, values=values
            )

    def test_incompatible_field_and_topology(self, field):
        """
        Test creating a Field with invalid shape.
        """
        topology = field.topology
        more = torch.tensor([3.0, 7.0, 6.0, -5.0, 4.0]).unsqueeze(-1)
        with pytest.raises(ValueError):
            wrong_field = Field(
                name="more field values than nodes", topology=topology, values=more
            )

        less = torch.tensor([3.0, 7.0, 6.0]).unsqueeze(-1)
        with pytest.raises(ValueError):
            wrong_field = Field(
                name="less field values than nodes", topology=topology, values=more
            )

    def test_full_values(self, field):
        """
        Test method Field.full_values()
        """
        expected_values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
        assert field.full_values() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_at_elements(self, field):
        """
        Test method Field.at_elements()
        """

        expected_values = torch.tensor([[3.0, 7.0], [7.0, 6.0], [6.0, -5.0]]).unsqueeze(
            -1
        )
        assert field.at_elements() == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )
