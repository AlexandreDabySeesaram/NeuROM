"""Standing guarantee: ``interpolate`` reproduces a function on *every* finite element.

For each element, interpolating a function of its exactness degree (constant / linear /
quadratic / cubic) must yield an FE field that equals that function at the quadrature points.
When a new element is added, add it to ``CASES`` here.
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.elements import (
    P1_BAR,
    P1_TRIANGLE,
    P2_TRIANGLE,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
    VectorElement,
)
from neurom.quadratures import TwoPoints1D, MidPoint2D
from neurom.interpolation import QuadratureContext, QuadratureAssembly, interpolate


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _bar_mesh(n=5):
    cells = torch.vstack([torch.arange(n - 1), torch.arange(1, n)]).T
    pts = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    topo = Topology(cells)
    coords = Field(FunctionSpace(topo, VectorElement(P1_BAR, 1)), pts)
    return Mesh(topo, coords), TwoPoints1D()


def _triangle_mesh():
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    topo = Topology(cells)
    coords = Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), verts)
    return Mesh(topo, coords), MidPoint2D()


# (id, element, mesh-builder, f: (n, gdim) -> (n[, n_comp]), that f's value shape)
CASES = [
    ("P1_BAR", P1_BAR, _bar_mesh, lambda X: 2 * X[:, 0] + 1.0),
    ("HERMITE", HERMITE, _bar_mesh, lambda X: X[:, 0] ** 3 - 2 * X[:, 0]),
    ("DG0_BAR", DG0_BAR, _bar_mesh, lambda X: torch.full((X.shape[0],), 3.0)),
    ("P1_TRIANGLE", P1_TRIANGLE, _triangle_mesh, lambda X: X[:, 0] + 2 * X[:, 1] + 1.0),
    (
        "P2_TRIANGLE",
        P2_TRIANGLE,
        _triangle_mesh,
        lambda X: X[:, 0] ** 2 + X[:, 0] * X[:, 1] + X[:, 1] ** 2,
    ),
    (
        "DG0_TRIANGLE",
        DG0_TRIANGLE,
        _triangle_mesh,
        lambda X: torch.full((X.shape[0],), -1.5),
    ),
    (
        "P1_TRIANGLE_vector",
        VectorElement(P1_TRIANGLE, 2),
        _triangle_mesh,
        lambda X: torch.stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]], dim=1),
    ),
]


@pytest.mark.parametrize("name, element, make_mesh, f", [(c[0], *c[1:]) for c in CASES])
def test_interpolation_reproduces_the_function(name, element, make_mesh, f):
    mesh, quad = make_mesh()
    space = FunctionSpace(mesh.topology, element)

    u = Field(space, interpolate(space, mesh, f), name=name)
    result = QuadratureAssembly(QuadratureContext(mesh, quad), u).interpolate()

    # The FE interpolant must equal f at the quadrature points (exact for its degree).
    x_q = result.x.values.reshape(-1, mesh.dim)
    f_at_q = f(x_q).reshape(result.u.values.shape)
    assert torch.allclose(result.u.values, f_at_q, atol=1e-12), name
