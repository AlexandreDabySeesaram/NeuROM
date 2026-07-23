import pytest
import torch

# Import library modules
from neurom.geometry.iso_parametric_mapping_1d import IsoparametricMapping1D
from neurom.meshes import Connectivity, Mesh
from neurom.fields import Field
from neurom.shape_functions.linear_bar import LinearBar
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

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
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    connectivity = Connectivity(nodes, elements)
    positions = Field(name="x", connectivity=connectivity, values=coords)
    mesh = Mesh(connectivity=connectivity, nodes_positions=positions)
    sf = LinearBar()
    field = Field(name="u", connectivity=connectivity, values=coords**2)
    return PointWiseInterpolator(mesh, sf, field, IsoparametricMapping1D(sf, mesh))


def test_at_position_interpolates_each_point_independently(interpolator):
    """Every query point is interpolated in its own element.

    Oracle is hand-computed linear interpolation of u = x**2 inside the element
    containing each point -- independent of the interpolator itself. Regression
    guard: a shape mistake that collapses the query onto a single point yields
    the correct output *shape*, so only an independent oracle catches it.
    """
    pts = torch.tensor([2.5, 5.0, 7.5, 1.0])

    out = interpolator.at_position(pts.reshape(-1, 1, 1))

    # Nodes at 0, 2.5, 5, 7.5, 10 -> 2.5, 5.0 and 7.5 are nodes (exact x**2);
    # 1.0 sits in element [0, 2.5] and interpolates to 0 + (6.25 - 0) * 1/2.5.
    expected = torch.tensor([6.25, 25.0, 56.25, 2.5])
    assert out.reshape(-1).detach().numpy() == pytest.approx(expected.numpy(), rel=1e-5)


@pytest.mark.parametrize("bad_shape", [(4,), (4, 1), (2, 2, 2, 1)])
def test_at_position_rejects_wrong_rank(interpolator, bad_shape):
    """Reject anything that is not (N_pts, N_q, dim).

    A flat (N_pts,) tensor is the dangerous case: it broadcasts inside
    ``inverse_map_at`` into a point-by-element cross product that the shape
    function slices back to the right shape, so it returns wrong numbers
    instead of raising.
    """
    x = torch.zeros(bad_shape)

    with pytest.raises(ValueError, match="expects x of shape"):
        interpolator.at_position(x)
