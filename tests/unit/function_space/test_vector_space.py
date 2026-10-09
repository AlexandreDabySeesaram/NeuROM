"""Vector-valued function spaces (P1 triangle, 2 components) and per-component BCs."""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.elements import VectorElement, P1_TRIANGLE
from neurom.function_space import FunctionSpace, DirichletBC
from neurom.interpolation import interpolate


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _square_mesh():
    """Unit square split into two triangles: vertices 0..3, cells [0,1,2],[0,2,3]."""
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    topology = Topology(cells)
    geometry = FunctionSpace(topology, VectorElement(P1_TRIANGLE, 2))
    mesh = Mesh(topology, Field(geometry, verts))
    return mesh, mesh.connectivity, verts


def test_vector_interpolate_sets_each_component():
    mesh, _, verts = _square_mesh()
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))
    # f(x, y) = (x, 2y): a different expression per component, one callable.
    init = interpolate(
        space, mesh, lambda X: torch.stack([X[:, 0], 2 * X[:, 1]], dim=1)
    )
    assert init.shape == (4, 2)
    assert torch.allclose(init[:, 0], verts[:, 0])  # u_x = x
    assert torch.allclose(init[:, 1], 2 * verts[:, 1])  # u_y = 2y


def test_vector_clamp_bc_sets_each_component_and_leaves_no_free_dofs():
    """A per-component vector clamp on every vertex fixes all DOFs to its values."""
    mesh, _, _ = _square_mesh()
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))

    bottom, top = [0, 1], [2, 3]  # fix u=(0,0) at bottom, u=(0,-1) at top
    u = TrainableField(
        space,
        0.1 * torch.ones(4, 2),
        bcs=[
            DirichletBC(0, bottom, value=[0.0, 0.0]),
            DirichletBC(0, top, value=[0.0, -1.0]),
        ],
        name="u",
    )
    # all 4 vertices clamped in both components -> no free DOFs
    assert u.values_reduced.numel() == 0
    full = u.full_values().detach()
    assert torch.allclose(full[bottom], torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    assert torch.allclose(full[top], torch.tensor([[0.0, -1.0], [0.0, -1.0]]))


def test_roller_bc_frees_the_unconstrained_component():
    """Fixing only u_x on a node leaves u_y trainable (a roller)."""
    mesh, _, _ = _square_mesh()
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))
    u = TrainableField(
        space,
        0.3 * torch.ones(4, 2),
        bcs=[DirichletBC(0, [0, 3], components=[0], value=0.0)],  # left edge: u_x = 0
        name="u",
    )
    # 4 nodes * 2 comp = 8 DOFs, 2 of them (u_x at nodes 0 and 3) fixed -> 6 free
    assert u.values_reduced.numel() == 6
    full = u.full_values().detach()
    assert full[0, 0] == 0.0 and full[3, 0] == 0.0  # u_x clamped
    assert full[0, 1] == pytest.approx(0.3) and full[3, 1] == pytest.approx(
        0.3
    )  # u_y free


def test_roller_gradient_flows_only_to_free_components():
    mesh, _, _ = _square_mesh()
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))
    u = TrainableField(
        space,
        torch.zeros(4, 2),
        bcs=[DirichletBC(0, [0], components=[0], value=0.0)],
        name="u",
    )
    u.full_values().sum().backward()
    # one free-DOF gradient per trainable component; the fixed u_x(node0) is absent
    assert u.values_reduced.grad.numel() == u.values_reduced.numel()
    assert torch.allclose(u.values_reduced.grad, torch.ones_like(u.values_reduced.grad))
