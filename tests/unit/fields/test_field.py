import pytest
import torch

# Import library modules
from neurom.fields import Field
from neurom.meshes import Mesh, Topology
from neurom.elements import P1_BAR, VectorElement
from neurom.function_space import FunctionSpace

torch.set_default_dtype(torch.float32)


@pytest.fixture
def field():
    """
    Prepare what is needed to define a Field:
    * name = "test"
    * P1 space over a 3-cell / 4-vertex bar mesh.
    * Values: [3., 7., 6., -5.]
    """
    N = 4
    cell_vertices = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, points, name="x")
    mesh = Mesh(topology, coords)
    space = FunctionSpace(mesh.topology, P1_BAR)
    values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    field = Field(space, values, name="test")

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
        space = field.space
        values = torch.tensor([3.0, 7.0, 6.0, -5.0])  # 1-D, missing component axis
        with pytest.raises(ValueError):
            Field(space, values, name="missing field dimension")

    def test_incompatible_field_and_connectivity(self, field):
        """
        Test creating a Field whose row count does not match the space's DOF count.
        """
        space = field.space
        more = torch.tensor([3.0, 7.0, 6.0, -5.0, 4.0]).unsqueeze(-1)
        with pytest.raises(ValueError):
            Field(space, more, name="more field values than dofs")

        less = torch.tensor([3.0, 7.0, 6.0]).unsqueeze(-1)
        with pytest.raises(ValueError):
            Field(space, less, name="less field values than dofs")

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
