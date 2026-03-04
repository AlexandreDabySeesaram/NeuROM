import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology, Mesh
from neurom.fields import Field, TrainableField
from neurom.constraints.no_constraint import NoConstraint
from neurom.shape_functions.linear_segment import LinearSegment
from neurom.interpolation.quadrature_interpolator import QuadratureInterpolator
from neurom.interpolation.quadrature_positions import QuadraturePositions

torch.set_default_dtype(torch.float32)


@pytest.fixture
def mesh():
    """
    Prepare what is needed to define a Mesh with positions as a Field:
    * Simple topology: 3 elements with 4 nodes.
    * Positions: [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    x = Field(name="x", topology=topology, values=values)
    mesh = Mesh(topology=topology, nodes_positions=x)

    return mesh


@pytest.fixture
def trainable_mesh():
    """
    Prepare what is needed to define a Mesh with positions as a TrainableField with NoConstraint:
    * Simple topology: 3 elements with 4 nodes.
    * Positions: [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    x = TrainableField(
        name="x", topology=topology, init_values=values, constraint=NoConstraint()
    )
    mesh = Mesh(topology=topology, nodes_positions=x)

    return mesh


class TestQuadratureInterpolator:
    """Test QuadratureInterpolator class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_quadrature_interpolator_with_mesh(self, mesh):
        """Test the QuadratureInterpolator class with a Mesh"""
        # Create shape function
        sf = LinearSegment()
        # Quadrature strategy
        quad = MidPoint1D()
        # Mapping from/to reference/physical coordinates
        mapping = IsoparametricMapping1D(sf)
        quad_interp = QuadratureInterpolator(
            mesh=mesh, quad=MidPoint1D(), mapping=IsoparametricMapping1D(sf)
        )

        # Prepare expected result
        expected = QuadraturePositions(
            xi_ref=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
            x_phys=torch.tensor([5.0, 6.5, 0.5])[:, None, None],
            xi_back=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
        )

        # Check computed result
        interp = quad_interp.interpolate()
        assert interp.xi_ref.detach() == pytest.approx(
            expected.xi_ref, rel=self.relative_tolerance
        )
        assert interp.x_phys.detach() == pytest.approx(
            expected.x_phys, rel=self.relative_tolerance
        )
        assert interp.xi_back.detach() == pytest.approx(
            expected.xi_back, rel=self.relative_tolerance
        )
