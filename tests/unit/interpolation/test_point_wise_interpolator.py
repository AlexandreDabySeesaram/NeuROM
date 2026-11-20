import pytest
import torch

# Import library modules
from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.interpolation import QuadratureContext, QuadratureAssembly
from neurom.quadratures import TwoPoints1D
from neurom.elements import P1_BAR, HERMITE, VectorElement

torch.set_default_dtype(torch.float32)


@pytest.fixture
def interpolator():
    """A 4-element mesh on [0, 10] carrying the nodal field u = x**2.

    A *non-linear* nodal field is deliberate: with a linear field, evaluating
    the wrong point in a neighbouring element still returns the right value by
    exact extrapolation, which hides indexing mistakes.
    """
    n = 5
    coords = torch.linspace(0.0, 10.0, n).unsqueeze(-1)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, coords))
    field = Field(FunctionSpace(mesh.topology, P1_BAR), coords**2, name="u")
    return PointWiseInterpolator(mesh, field)


def test_at_position_interpolates_each_point_independently(interpolator):
    """Every query point is interpolated in its own element.

    Oracle is hand-computed linear interpolation of u = x**2 inside the element
    containing each point -- independent of the interpolator itself. Regression
    guard: a shape mistake that collapses the query onto a single point yields
    the correct output *shape*, so only an independent oracle catches it.
    """
    pts = torch.tensor([2.5, 5.0, 7.5, 1.0])

    out = interpolator.at_position(pts.reshape(-1, 1))

    assert out.shape == (4, 1)

    # Nodes at 0, 2.5, 5, 7.5, 10 -> 2.5, 5.0 and 7.5 are nodes (exact x**2);
    # 1.0 sits in element [0, 2.5] and interpolates to 0 + (6.25 - 0) * 1/2.5.
    expected = torch.tensor([6.25, 25.0, 56.25, 2.5])
    assert out.reshape(-1).detach().numpy() == pytest.approx(expected.numpy(), rel=1e-5)


@pytest.mark.parametrize("bad_shape", [(4,), (4, 1, 1), (2, 2, 2, 1)])
def test_at_position_rejects_wrong_rank(interpolator, bad_shape):
    """Reject anything that is not (N_pts, dim).

    A flat (N_pts,) tensor is the dangerous case: it broadcasts inside
    ``inverse_map_at`` into a point-by-element cross product that the shape
    function slices back to the right shape, so it returns wrong numbers
    instead of raising.
    """
    x = torch.zeros(bad_shape)

    with pytest.raises(ValueError, match="expects x of shape"):
        interpolator.at_position(x)


def test_at_position_uses_the_field_connectivity():
    """Field values are gathered through the field's connectivity, not the mesh's.

    The field numbers its nodes in reverse order of the mesh, so reading its
    values through the mesh connectivity would return the mirrored field.
    """
    n = 5
    coords = torch.linspace(0.0, 10.0, n).unsqueeze(-1)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, coords))

    # Field node k sits at mesh node n - 1 - k: the field lives on its own space,
    # whose topology reverses the vertex numbering. The interpolator must gather
    # field values through the field's own connectivity (its space's DOF map),
    # not the mesh's -- reading through the mesh would return the mirrored field.
    field_space = FunctionSpace(Topology(n - 1 - elements), P1_BAR)
    field = Field(field_space, coords.flip(0) ** 2, name="u")

    interpolator = PointWiseInterpolator(mesh, field)

    out = interpolator.at_position(torch.tensor([[1.0], [7.5]]))

    # Same oracle as above: linear interpolation of u = x**2
    expected = torch.tensor([2.5, 56.25])
    assert out.reshape(-1).detach().numpy() == pytest.approx(expected.numpy(), rel=1e-5)


def _cubic_hermite_on_non_uniform_mesh():
    """Hermite field w = x**3 with physical DOFs (w, w') on the mesh [0, 0.25, 1]."""
    x_nodes = torch.tensor([0.0, 0.25, 1.0])
    n = len(x_nodes)
    elements = torch.vstack([torch.arange(n - 1), torch.arange(1, n)]).T
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, x_nodes.unsqueeze(-1)))

    # Hermite DOFs interleaved per vertex as (w, w'): the Hermite element's DOF
    # layout numbers them [w0, w'0, w1, w'1, ...], matching this values tensor.
    values = torch.empty(2 * n, 1)
    values[0::2, 0] = x_nodes**3
    values[1::2, 0] = 3 * x_nodes**2
    w = Field(FunctionSpace(mesh.topology, HERMITE), values, name="w")
    return mesh, w


def test_at_position_hermite_reproduces_cubic_on_non_uniform_mesh():
    """Hermite with physical slope DOFs reproduces w = x**3 exactly in both elements."""
    mesh, w = _cubic_hermite_on_non_uniform_mesh()
    interpolator = PointWiseInterpolator(mesh, w)

    pts = torch.tensor([0.1, 0.2, 0.4, 0.8])

    out = interpolator.at_position(pts.reshape(-1, 1))

    assert out.reshape(-1).detach().numpy() == pytest.approx((pts**3).numpy(), rel=1e-5)


def test_at_position_matches_quadrature_assembly():
    """Point-wise and quadrature interpolation agree at the quadrature points."""
    mesh, w = _cubic_hermite_on_non_uniform_mesh()
    ctx = QuadratureContext(mesh, TwoPoints1D())
    result = QuadratureAssembly(ctx, w).interpolate()

    x_q = result.x.values.detach().reshape(-1, 1)
    out = PointWiseInterpolator(mesh, w).at_position(x_q)

    assert out.reshape(-1).detach().numpy() == pytest.approx(
        result.u.values.reshape(-1).detach().numpy(), rel=1e-5
    )
