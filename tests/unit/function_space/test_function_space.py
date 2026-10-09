import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.quadratures import TwoPoints1D
from neurom.interpolation import QuadratureContext, QuadratureAssembly, interpolate
from neurom.elements import HERMITE, DG0_BAR, P1_BAR, VectorElement
from neurom.function_space import FunctionSpace, DirichletBC


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _mesh(N=5):
    cell_vertices = torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, points))
    return mesh, mesh.connectivity, mesh.coordinates


def test_function_space_takes_no_quadrature():
    mesh, _, _ = _mesh(5)
    space = FunctionSpace(mesh.topology, HERMITE)  # (mesh, element) only
    assert space.n_scalar_dofs == 2 * 5
    assert not hasattr(space, "quad") and not hasattr(space, "context")


def test_hermite_space_matches_hand_built_interpolation():
    N = 5
    mesh, _, x = _mesh(N)
    ctx = QuadratureContext(mesh, TwoPoints1D())

    x_n = x.values.squeeze(-1)
    w0 = 1e-2 * (1 - torch.cos(2 * torch.pi * x_n))
    slope0 = 1e-2 * 2 * torch.pi * torch.sin(2 * torch.pi * x_n)

    space = FunctionSpace(mesh.topology, HERMITE)

    # --- hand-built: DOF vector assembled by hand, BCs located by explicit DOF ---
    # Hermite global DOFs interleave (value_v, slope_v) per vertex v.
    init_old = torch.stack([w0, slope0], dim=1).reshape(-1, 1)
    u_old = TrainableField(
        space,
        init_old,
        bcs=[DirichletBC(0, [0, N - 1], kinds=["value", "d1"])],
        name="u_old",
    )
    r_old = QuadratureAssembly(ctx, u_old).interpolate()

    # --- FunctionSpace: element in, init seeded by interpolate(f), BCs by meaning ---
    init_new = interpolate(
        space, mesh, lambda z: 1e-2 * (1 - torch.cos(2 * torch.pi * z)).squeeze(-1)
    )
    u_new = TrainableField(
        space,
        init_new,
        bcs=[DirichletBC(0, [0, N - 1], kinds=["value", "d1"])],
        name="u_new",
    )
    r_new = QuadratureAssembly(ctx, u_new).interpolate()

    assert torch.allclose(r_old.u.values, r_new.u.values, atol=1e-12)
    assert torch.allclose(r_old.x.values, r_new.x.values, atol=1e-14)
    assert u_new.values_reduced.numel() == 2 * N - 4  # ends clamped


def test_interpolate_sets_value_and_derivative_dofs():
    N = 5
    mesh, _, _ = _mesh(N)
    space = FunctionSpace(mesh.topology, HERMITE)
    # f = x^3 -> value DOFs x^3, slope DOFs 3x^2
    init = interpolate(space, mesh, lambda z: (z**3).squeeze(-1))
    xn = torch.linspace(0, 1, N)
    assert torch.allclose(init[space.locate(0, range(N), ["value"])].squeeze(-1), xn**3)
    assert torch.allclose(
        init[space.locate(0, range(N), ["d1"])].squeeze(-1), 3 * xn**2
    )


def test_interpolate_raises_for_a_dof_kind_without_a_functional():
    # interpolate is honestly limited to point-functional elements; a flux/normal-derivative
    # kind has no functional yet, so it raises -- and field init never depends on it.
    from neurom.elements import FiniteElement, NORMAL_DERIVATIVE, EntityDim
    from neurom.shape_functions import LinearBar

    mesh, _, _ = _mesh(3)
    exotic = FiniteElement(
        LinearBar(),
        [
            (EntityDim.VERTEX, 0, NORMAL_DERIVATIVE),
            (EntityDim.VERTEX, 1, NORMAL_DERIVATIVE),
        ],
    )
    space = FunctionSpace(mesh.topology, exotic)
    with pytest.raises(NotImplementedError, match="no functional"):
        interpolate(space, mesh, lambda z: z.squeeze(-1))


def test_dg0_space_interpolates_constant_per_cell():
    N = 5
    mesh, _, _ = _mesh(N)
    ctx = QuadratureContext(mesh, TwoPoints1D())
    space = FunctionSpace(mesh.topology, DG0_BAR)
    vals = (
        torch.arange(1, N).unsqueeze(-1).to(torch.get_default_dtype())
    )  # (n_cells, 1)
    c = TrainableField(space, vals, name="c")
    r = QuadratureAssembly(ctx, c).interpolate()
    expected = vals.unsqueeze(1).expand(-1, r.u.values.shape[1], -1)
    assert torch.allclose(r.u.values, expected, atol=1e-14)


def test_p1_space_uses_identity_and_vertex_dofs():
    N = 5
    mesh, _, _ = _mesh(N)
    ctx = QuadratureContext(mesh, TwoPoints1D())
    space = FunctionSpace(mesh.topology, P1_BAR)
    assert space.n_scalar_dofs == N
    u = TrainableField(space, torch.linspace(0, 1, N).unsqueeze(-1), name="u")
    r = QuadratureAssembly(ctx, u).interpolate()
    # interpolation of a linear nodal field is exact at quadrature points
    assert torch.allclose(r.u.values.squeeze(-1), r.x.values.squeeze(-1), atol=1e-12)
