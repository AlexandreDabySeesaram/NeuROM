import pytest
import torch

# Import library modules
from neurom.meshes.topology import Topology
from neurom.fields import Field, TrainableField
from neurom.constraints.no_constraint import NoConstraint
from neurom.shape_functions.linear_segment import LinearSegment
from neurom.interpolation.field_interpolator import FieldInterpolator

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
    values = torch.tensor([3.0, 7.0, 6.0, -5.0])
    field = Field(name="test", topology=topology, values=values)

    return field


class TestFieldInterpolator:
    """Test FieldInterpolator class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_field_interpolator_with_field(self, field):
        """Test the FieldInterpolator class with a Field"""
        # Create FieldInterpolator
        field_interp = FieldInterpolator(sf=LinearSegment(), field=field)

        # Prepare reference positions, tensor of shape (N_e, N_q, dim) and expected values
        # Midpoints of values
        values = [  # Midpoint
            {
                "xi": torch.tensor([0.0, 0.0, 0.0])[:, None, None],
                "expected": torch.tensor([5.0, 6.5, 0.5]).unsqueeze(-1),
            },
            # Start
            {
                "xi": torch.tensor([-1.0, -1.0, -1.0])[:, None, None],
                "expected": torch.tensor([3.0, 7.0, 6.0]).unsqueeze(-1),
            },
            # End
            {
                "xi": torch.tensor([1.0, 1.0, 1.0])[:, None, None],
                "expected": torch.tensor([7.0, 6.0, -5.0]).unsqueeze(-1),
            },
            # Mix
            {
                "xi": torch.tensor([0.5, -0.75, -0.5])[:, None, None],
                "expected": torch.tensor([6.0, 6.875, 3.25]).unsqueeze(-1),
            },
        ]

        for val in values:
            xi = val["xi"]
            expected = val["expected"]
            interp = field_interp.at_reference(xi)
            assert interp == pytest.approx(expected, rel=self.relative_tolerance), (
                "xi = ",
                xi,
            )
