import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.quadratures.mid_point_1d import MidPoint1D
from neurom.geometry.iso_parametric_mapping_1d import IsoparametricMapping1D
from neurom.meshes import Connectivity, Mesh
from neurom.fields import Field, TrainableField
from neurom.constraints.no_constraint import NoConstraint
from neurom.shape_functions.linear_bar import LinearBar
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.interpolation.quadrature_positions import QuadraturePositions

torch.set_default_dtype(torch.float32)


@pytest.fixture
def mesh():
    """
    Prepare what is needed to define a Mesh with positions as a Field:
    * Simple connectivity: 3 elements with 4 nodes.
    * Positions: [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    connectivity = Connectivity(nodes, elements)
    values = torch.tensor([3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
    x = Field(name="x", connectivity=connectivity, values=values)
    mesh = Mesh(connectivity=connectivity, nodes_positions=x)

    return mesh


class TestQuadratureContext:
    """Test QuadratureContext class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_quadrature_context(self, mesh):
        """Test the QuadratureContext class with a Mesh"""
        quad_context = QuadratureContext(
            mesh=mesh,
            quad=MidPoint1D(),
            mapping=IsoparametricMapping1D(
                LinearBar(),
                mesh,
            ),
        )

        # Prepare expected interpolated positiosn
        interp_expected = QuadraturePositions(
            xi_ref=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
            x_phys=torch.tensor([5.0, 6.5, 0.5])[:, None, None],
            xi_back=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
        )

        # Check computed interpolation
        interp = quad_context.interpolate
        assert interp.xi_ref.values.detach() == pytest.approx(
            interp_expected.xi_ref.detach(), rel=self.relative_tolerance
        )
        assert interp.x_phys.values.detach() == pytest.approx(
            interp_expected.x_phys, rel=self.relative_tolerance
        )
        assert interp.xi_back.values.detach() == pytest.approx(
            interp_expected.xi_back, rel=self.relative_tolerance
        )

        # Check measure
        measure = quad_context.measure
        # Weight = 0.5 and reference measure = 2.
        measure_expected = torch.tensor([4.0, 1.0, 11.0]).reshape(3, 1, 1)
        assert measure.values == pytest.approx(
            measure_expected, rel=self.relative_tolerance
        )

        # Change positions
        new_values = torch.tensor([2.0, 5.0, 15.0, -10.0]).unsqueeze(-1)
        mesh.nodes_positions = Field(
            name="x", connectivity=mesh.connectivity, values=new_values
        )

        # Check the interpolation and the measure did not change
        interp = quad_context.interpolate
        assert interp.xi_ref.values.detach() == pytest.approx(
            interp_expected.xi_ref, rel=self.relative_tolerance
        )
        assert interp.x_phys.values.detach() == pytest.approx(
            interp_expected.x_phys, rel=self.relative_tolerance
        )
        assert interp.xi_back.values.detach() == pytest.approx(
            interp_expected.xi_back, rel=self.relative_tolerance
        )

        measure = quad_context.measure
        assert measure.values == pytest.approx(
            measure_expected, rel=self.relative_tolerance
        )

        # Call update() and check the interpolation and the measure are now properly computed
        quad_context.update()

        # Check the interpolation and the measure did change
        interp = quad_context.interpolate
        interp_expected = QuadraturePositions(
            xi_ref=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
            x_phys=torch.tensor([3.5, 10.0, 2.5])[:, None, None],
            xi_back=torch.tensor([0.0, 0.0, 0.0])[:, None, None],
        )
        assert interp.xi_ref.values.detach() == pytest.approx(
            interp_expected.xi_ref, rel=self.relative_tolerance
        )
        assert interp.x_phys.values.detach() == pytest.approx(
            interp_expected.x_phys, rel=self.relative_tolerance
        )
        assert interp.xi_back.values.detach() == pytest.approx(
            interp_expected.xi_back, rel=self.relative_tolerance
        )

        measure = quad_context.measure
        # Weight = 0.5 and reference measure = 2.
        measure_expected = torch.tensor([3.0, 10.0, 25.0]).reshape(3, 1, 1)
        assert measure.values == pytest.approx(
            measure_expected, rel=self.relative_tolerance
        )
