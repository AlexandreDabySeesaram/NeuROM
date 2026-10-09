"""DG0 via the FunctionSpace API: a piecewise-constant field on a DG0 element."""

import pytest
import torch

from neurom.interpolation import (
    PointWiseInterpolator,
    QuadratureContext,
    QuadratureAssembly,
)
from neurom.math import jacobian, integrate
from neurom.meshes import Mesh, Topology
from neurom.fields import TrainableField, Field
from neurom.quadratures import TwoPoints1D
from neurom.elements import DG0_BAR, P1_BAR, VectorElement
from neurom.function_space import FunctionSpace


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _space(N=4):
    cell_vertices = torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, points)
    mesh = Mesh(topology, coords)
    ctx = QuadratureContext(mesh, TwoPoints1D())
    return FunctionSpace(mesh.topology, DG0_BAR), ctx, mesh


def test_dg0_has_one_dof_per_cell():
    space, _, _ = _space(N=4)
    assert space.n_scalar_dofs == 3  # 3 cells
    c = TrainableField(space, torch.tensor([[1.0], [2.0], [3.0]]), name="c")
    assert c.at_elements().shape == (3, 1, 1)  # (n_cells, n_dof=1, dim)


def test_dg0_interpolates_to_constant_per_element():
    space, ctx, _ = _space(N=4)
    vals = torch.tensor([[10.0], [20.0], [30.0]])
    c = TrainableField(space, vals, name="c")
    res = QuadratureAssembly(ctx, c).interpolate()
    expected = vals.unsqueeze(1).expand(-1, res.u.values.shape[1], -1)
    assert torch.allclose(res.u.values, expected, atol=1e-14)


def test_dg0_jacobian_is_zero():
    space, ctx, _ = _space(N=4)
    c = TrainableField(space, torch.tensor([[10.0], [20.0], [30.0]]), name="c")
    res = QuadratureAssembly(ctx, c).interpolate()
    du = jacobian(res.x, res.u)  # constant in space -> zeros, must not raise
    assert torch.allclose(du.values, torch.zeros_like(du.values))


def test_dg0_energy_is_differentiable_in_cell_dofs():
    # E(c) = integrate(c dx) = sum_e c_e |Omega_e|; dE/dc_e = |Omega_e| = 1/3 each.
    space, ctx, _ = _space(N=4)
    c = TrainableField(space, torch.tensor([[1.0], [2.0], [3.0]]), name="c")
    res = QuadratureAssembly(ctx, c).interpolate()
    integrate(res.u.values * res.measure.values).backward()
    grad = c.values_reduced.grad
    assert torch.allclose(grad, torch.full_like(grad, 1.0 / 3.0), atol=1e-12)


def test_dg0_pointwise_interpolation_returns_the_containing_cell_value():
    space, _, mesh = _space(N=5)  # 4 cells on [0, 1]
    vals = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    c = TrainableField(space, vals, name="c")
    interp = PointWiseInterpolator(mesh, c)
    out = interp.at_position(torch.tensor([[0.1], [0.3], [0.6], [0.9]]))
    assert torch.allclose(out, vals)
