"""WP7: boundary entities, closure in locate/BCs, and callable Dirichlet values.

- T-WP7.1 boundary counts (unit square: 4n edges, 4n vertices; 1-D bar: 2 end vertices);
- T-WP7.2 a P2 selection of boundary edges with closure equals the edge-plus-vertex selection;
- T-WP7.3 a callable Dirichlet value reproduces an affine (P1) / quadratic (P2) field exactly
  (patch test): fixing every DOF through the functionals yields the FE interpolant of f, which
  point-evaluates back to f;
- T-WP7.4 constant BCs are unchanged (covered by the existing BC tests + a direct mask check here).
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace, DirichletBC
from neurom.fields import Field
from neurom.elements import P1_TRIANGLE, P2_TRIANGLE, VectorElement, EntityDim
from neurom.interpolation import PointWiseInterpolator, interpolate


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _square(n):
    """Structured unit-square triangle mesh (alternating diagonals): topology + mesh."""
    xs, ys = torch.meshgrid(
        torch.linspace(0, 1, n + 1), torch.linspace(0, 1, n + 1), indexing="ij"
    )
    pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    vid = lambda i, j: i * (n + 1) + j
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    topo = Topology(torch.tensor(cells))
    mesh = Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))
    return topo, mesh


def test_closure_of_boundary_edges_adds_their_vertices():
    """(T-WP7.2) locating boundary edges with closure equals edge DOFs + their vertices' DOFs."""
    topo, _ = _square(3)
    space = FunctionSpace(topo, P2_TRIANGLE)
    bedges = topo.boundary(EntityDim.EDGE).tolist()
    bverts = topo.boundary(EntityDim.VERTEX).tolist()

    with_closure = space.locate(EntityDim.EDGE, bedges, closure=True)
    manual = torch.unique(
        torch.cat(
            [
                space.locate(EntityDim.EDGE, bedges),
                space.locate(EntityDim.VERTEX, bverts),
            ]
        )
    )
    assert torch.equal(with_closure, manual)
    # closure is a strict superset of the edge-only selection
    assert with_closure.numel() > space.locate(EntityDim.EDGE, bedges).numel()


def test_constant_bc_mask_is_unchanged_by_the_refactor():
    """(T-WP7.4) a plain constant clamp still fixes exactly its DOFs, every component."""
    topo, _ = _square(2)
    space = FunctionSpace(topo, P1_TRIANGLE)
    bverts = topo.boundary(EntityDim.VERTEX).tolist()
    free_mask, imposed = space.compile_constraint(
        [DirichletBC(EntityDim.VERTEX, bverts, value=0.0)], 1
    )
    fixed = ~free_mask
    assert torch.equal(
        fixed.squeeze(-1).nonzero().reshape(-1),
        space.locate(EntityDim.VERTEX, bverts),
    )
    assert torch.equal(imposed, torch.zeros_like(imposed))


def _reproduces(space, mesh, bcs, f):
    """Fix every DOF to the interpolant of f, then check the field reproduces f at interior points.

    Geometry enters only through ``interpolate(space, mesh, f)`` (mesh-explicit); the resulting
    per-DOF tensor is the BC value, so ``compile_constraint`` stays geometry-free.
    """
    free_mask, imposed = space.compile_constraint(bcs, 1)
    assert not free_mask.any()  # every DOF fixed
    pts = torch.tensor([[0.3, 0.4], [0.6, 0.2], [0.5, 0.55]])  # f64 under the fixture
    out = PointWiseInterpolator(mesh, Field(space, imposed)).at_position(pts)
    assert out.detach().numpy() == pytest.approx(f(pts).numpy(), rel=1e-10)


def test_function_valued_dirichlet_reproduces_affine_on_p1():
    """(T-WP7.3) imposing interpolate(f) on every P1 vertex reproduces an affine field exactly."""
    topo, mesh = _square(3)
    space = FunctionSpace(topo, P1_TRIANGLE)
    f = lambda X: 2.0 - X[:, 0:1] + 0.5 * X[:, 1:2]  # affine, lives in P1
    g = interpolate(space, mesh, f)  # (n_scalar_dofs, 1) -- the one geometry-aware step
    verts = list(range(topo.n_vertices))
    _reproduces(space, mesh, [DirichletBC(EntityDim.VERTEX, verts, value=g)], f)


def test_function_valued_dirichlet_reproduces_quadratic_on_p2():
    """(T-WP7.3) imposing interpolate(f) on every P2 DOF (all edges + closure) reproduces a quadratic."""
    topo, mesh = _square(3)
    space = FunctionSpace(topo, P2_TRIANGLE)
    # quadratic, lives in P2
    f = lambda X: 1.0 + 2 * X[:, 0:1] - X[:, 1:2] + 3 * X[:, 0:1] ** 2 + X[:, 0:1] * X[:, 1:2]
    g = interpolate(space, mesh, f)
    all_edges = list(range(topo.n_entities(EntityDim.EDGE)))  # closure also covers every vertex
    _reproduces(space, mesh, [DirichletBC(EntityDim.EDGE, all_edges, value=g, closure=True)], f)


def test_callable_dirichlet_value_is_rejected():
    """A callable value is rejected: FunctionSpace is geometry-free, so resolve it via interpolate."""
    topo, _ = _square(2)
    space = FunctionSpace(topo, P1_TRIANGLE)
    bc = DirichletBC(EntityDim.VERTEX, [0], value=lambda X: X[:, 0:1])
    with pytest.raises(TypeError, match="geometry-free"):
        space.compile_constraint([bc], 1)
