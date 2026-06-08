"""Tests of the 2D isoparametric mapping through the quadrature pipeline.

A unit square split into two triangles is used to verify that the
geometry-dependent quantities (Jacobian determinant, quadrature measure,
physical positions) are consistent: integrating the constant 1 over the mesh
must give back the mesh area exactly, for every 2D quadrature rule.
"""

import pytest
import torch

from neurom.quadratures import MidPoint2D, ThreePoints2D
from neurom.shape_functions import LinearTriangle
from neurom.geometry import IsoparametricMapping2D
from neurom.meshes import Mesh, Connectivity
from neurom.fields import Field
from neurom.interpolation import QuadratureContext

torch.set_default_dtype(torch.float32)

relative_tolerance = 1e-6


@pytest.fixture
def unit_square_mesh():
    """
    Unit square [0,1]^2 split into two CCW triangles.

        3 ---- 2
        | \\  |
        |  \\ |
        0 ---- 1
    """
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # (4, 2)
    nodes = torch.arange(0, 4)
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]])

    connectivity = Connectivity(nodes, elements)
    x = Field(name="positions", connectivity=connectivity, values=positions)
    return Mesh(connectivity=connectivity, nodes_positions=x)


@pytest.mark.parametrize("quad", [MidPoint2D(), ThreePoints2D()])
def test_measure_sums_to_mesh_area(unit_square_mesh, quad):
    """
    Test that the quadrature measure integrates the constant 1 exactly.

    Sum of weight * |det_J| over all elements and quadrature points must be
    the total mesh area (1 for the unit square).
    """
    sf = LinearTriangle()
    mapping = IsoparametricMapping2D(sf, unit_square_mesh)
    ctx = QuadratureContext(unit_square_mesh, quad, mapping)

    area = ctx.measure.values.sum().item()
    assert area == pytest.approx(1.0, rel=relative_tolerance)


@pytest.mark.parametrize("quad", [MidPoint2D(), ThreePoints2D()])
def test_quadrature_positions_round_trip(unit_square_mesh, quad):
    """
    Test that mapping reference -> physical -> reference is the identity.
    """
    sf = LinearTriangle()
    mapping = IsoparametricMapping2D(sf, unit_square_mesh)
    ctx = QuadratureContext(unit_square_mesh, quad, mapping)

    xi_ref = ctx.interpolate.xi_ref.values
    xi_back = ctx.interpolate.xi_back.values

    assert xi_back.detach().numpy() == pytest.approx(
        xi_ref.detach().numpy(), rel=relative_tolerance
    )


def test_map_reproduces_vertices(unit_square_mesh):
    """
    Test that mapping the reference simplex vertices gives the element nodes.
    """
    sf = LinearTriangle()
    mapping = IsoparametricMapping2D(sf, unit_square_mesh)

    # Reference vertices as one "quadrature" set per element: (N_e, 3, 2)
    xi_vertices = sf.reference_element.simplex.unsqueeze(0).expand(2, -1, -1)
    x = mapping.map(xi_vertices)

    expected = unit_square_mesh.nodes_positions.at_elements()
    assert x.detach().numpy() == pytest.approx(
        expected.detach().numpy(), rel=relative_tolerance
    )
