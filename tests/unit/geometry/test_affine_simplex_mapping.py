"""Tests for the affine reference-to-physical mapping (bar and triangle).

``AffineSimplexMapping`` is dimension-generic and fully analytical (no autograd): these tests cover
the forward/inverse maps, the Jacobian determinant and ``jacobian_at`` (shape, value, element subset,
and differentiability w.r.t. node positions) in 1-D, and the quadrature-measure/round-trip/vertex
reproduction in 2-D.
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.elements import P1_BAR, P1_TRIANGLE, VectorElement
from neurom.fields import Field
from neurom.geometry import AffineSimplexMapping
from neurom.samplings import QuadratureSampling
from neurom.shape_functions import LinearTriangle
from neurom.quadratures import MidPoint2D, ThreePoints2D
from neurom.interpolation import QuadratureContext

torch.set_default_dtype(torch.float32)

REL = 1e-6


# --------------------------------------------------------------------------------------------------
# 1-D (bar)
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def bar_mapping():
    """Single bar element [5, 10]."""
    elements = torch.tensor([0, 1]).reshape(1, 2)
    values = torch.tensor([5.0, 10.0]).reshape(2, 1)
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, values))
    return AffineSimplexMapping(mesh)


def test_map_reference_to_physical(bar_mapping):
    """xi in [-1, 1] maps affinely onto the physical bar [5, 10]."""
    xi = torch.tensor([[-1, -0.5, 0, 0.5, 1]]).unsqueeze(-1)  # (1, 5, 1)
    x = bar_mapping.map(QuadratureSampling(xi)).values
    x_expected = torch.tensor([[5, 6.25, 7.5, 8.75, 10.0]]).unsqueeze(-1)
    assert x == pytest.approx(x_expected, rel=REL)


def test_inverse_map_physical_to_reference(bar_mapping):
    """Physical positions map back to xi in [-1, 1] (xi = xi0 + J^{-1}(x - x0))."""
    x = torch.tensor([[5, 6.25, 7.5, 8.75, 10.0]]).unsqueeze(-1)
    xi = bar_mapping.inverse_map(QuadratureSampling(x)).values
    xi_expected = torch.tensor([[-1, -0.5, 0, 0.5, 1]]).unsqueeze(-1)
    assert xi == pytest.approx(xi_expected, rel=REL)


def test_det_jacobian(bar_mapping):
    """det J = (x1 - x0) / 2 = 2.5, shape (N_e, 1)."""
    det_J = bar_mapping.det_jacobian.values
    assert det_J.shape == (1, 1)
    assert det_J == pytest.approx(torch.tensor([2.5]).unsqueeze(-1), rel=REL)


def _non_uniform_bar(positions):
    """Two-element bar on the given node positions."""
    elements = torch.tensor([[0, 1], [1, 2]])
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, positions.reshape(3, 1)))
    return AffineSimplexMapping(mesh)


def test_jacobian_is_half_element_length():
    """J = h/2 per element, constant, broadcast over every reference point."""
    mapping = _non_uniform_bar(torch.tensor([0.0, 0.25, 1.0]))
    xi = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1).expand(2, 2, 1)
    J = mapping.jacobian_at(QuadratureSampling(xi)).values
    J_expected = torch.tensor([0.125, 0.375]).reshape(2, 1, 1, 1).expand(2, 2, 1, 1)
    assert J.shape == (2, 2, 1, 1)
    assert J == pytest.approx(J_expected, rel=REL)


def test_jacobian_on_element_subset():
    """element_ids restricts the computation to the selected elements."""
    mapping = _non_uniform_bar(torch.tensor([0.0, 0.25, 1.0]))
    xi = torch.zeros(1, 1, 1)
    J = mapping.jacobian_at(QuadratureSampling(xi), element_ids=torch.tensor([1])).values
    assert J.shape == (1, 1, 1, 1)
    assert J.item() == pytest.approx(0.375, rel=REL)


def test_inverse_map_at_round_trips_a_subset():
    """inverse_map_at recovers reference coords for the selected elements only."""
    mapping = _non_uniform_bar(torch.tensor([0.0, 0.25, 1.0]))  # 2 elements
    xi = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 3, 1).expand(2, 3, 1)  # (2, 3, 1)
    x = mapping.map(QuadratureSampling(xi)).values  # (2, 3, 1) physical
    ids = torch.tensor([1])
    xi_back = mapping.inverse_map_at(QuadratureSampling(x[ids]), ids).values  # (1, 3, 1)
    assert xi_back == pytest.approx(xi[ids], rel=REL)


def test_jacobian_is_differentiable_wrt_nodes():
    """Gradients flow from the (analytical) J back to the node positions."""
    xi = torch.tensor([-1.0, 0.3, 1.0], dtype=torch.float64).reshape(1, 3, 1).expand(2, 3, 1)

    def J_of_nodes(positions):
        return _non_uniform_bar(positions).jacobian_at(QuadratureSampling(xi)).values

    positions = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(J_of_nodes, (positions,))


# --------------------------------------------------------------------------------------------------
# 2-D (triangle)
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def unit_square_mesh():
    """Unit square [0,1]^2 split into two CCW triangles."""
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_TRIANGLE, 2))
    return Mesh(topology, Field(geometry, positions))


@pytest.mark.parametrize("quad", [MidPoint2D(), ThreePoints2D()])
def test_measure_sums_to_mesh_area(unit_square_mesh, quad):
    """Sum of weight * |det J| over all elements/quad points is the mesh area (1.0)."""
    ctx = QuadratureContext(unit_square_mesh, quad)
    area = ctx.measure.values.sum().item()
    assert area == pytest.approx(1.0, rel=REL)


@pytest.mark.parametrize("quad", [MidPoint2D(), ThreePoints2D()])
def test_quadrature_positions_round_trip(unit_square_mesh, quad):
    """reference -> physical -> reference is the identity."""
    ctx = QuadratureContext(unit_square_mesh, quad)
    xi_ref = ctx.interpolate.xi_ref.values
    xi_back = ctx.interpolate.xi_back.values
    assert xi_back.detach().numpy() == pytest.approx(xi_ref.detach().numpy(), rel=REL)


def test_map_reproduces_vertices(unit_square_mesh):
    """Mapping the reference simplex vertices gives the physical element nodes."""
    mapping = AffineSimplexMapping(unit_square_mesh)
    xi_vertices = LinearTriangle().reference_cell.simplex.unsqueeze(0).expand(2, -1, -1)
    x = mapping.map(QuadratureSampling(xi_vertices)).values
    expected = unit_square_mesh.coordinates.at_elements()
    assert x.detach().numpy() == pytest.approx(expected.detach().numpy(), rel=REL)
