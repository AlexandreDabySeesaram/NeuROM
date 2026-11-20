"""The 2-D vector FunctionSpace drives the elasticity physics end to end.

`examples/2d` was ported to ``FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))`` with
essential BCs declared *by meaning* via :class:`DirichletBC`. Here we drive the same
physics (SolidElasticEnergy with a Green-Lagrange strain and a linear-elastic stress) on a
hand-built two-triangle mesh and check the resulting potential energy against its known
value -- the example needs gmsh to run, this does not.
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.quadratures import MidPoint2D
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.field_layout import FieldLayout
from neurom.physics import SolidElasticEnergy
from neurom.physics.tensors import green_lagrange_strain, linear_elastic_stress_point
from neurom.physics_loss import PhysicsLoss
from neurom.elements import VectorElement, P1_TRIANGLE
from neurom.function_space import FunctionSpace, DirichletBC


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


LAME_LAMBDA, LAME_MU = 1.25, 1.0
VERTS = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
)
CELLS = torch.tensor([[0, 1, 2], [0, 2, 3]])
BOTTOM, TOP = (
    [0, 1],
    [2, 3],
)  # clamp bottom to (0,0), top to (0,-1) -- as in examples/2d


def _stress(strain):
    return linear_elastic_stress_point(strain, LAME_LAMBDA, LAME_MU)


def _energy():
    u_init = 0.1 * torch.ones(4, 2)
    fl = FieldLayout()
    topology = Topology(CELLS)
    geometry = FunctionSpace(topology, VectorElement(P1_TRIANGLE, 2))
    coords = Field(geometry, VERTS, name="positions")
    mesh = Mesh(topology, coords)
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))
    ctx = QuadratureContext(mesh, MidPoint2D())

    u = fl.add(
        TrainableField(
            space,
            u_init,
            bcs=[
                DirichletBC(0, BOTTOM, value=[0.0, 0.0]),
                DirichletBC(0, TOP, value=[0.0, -1.0]),
            ],
            name="displacement",
        )
    )

    asm = QuadratureAssembly(ctx, u)
    IntegrationDomain([asm]).interpolate_all(fl)
    physics = SolidElasticEnergy(
        field=u, strain=green_lagrange_strain, stress_point=_stress
    )
    return PhysicsLoss(physics=physics, field_layout=fl)()


def test_vector_space_energy_matches_reference():
    assert _energy().item() == pytest.approx(1.625, rel=1e-9)


def test_new_vector_space_free_dof_count():
    # 4 vertices x 2 components = 8 scalar DOFs; all 4 vertices clamped -> 0 free.
    topology = Topology(CELLS)
    geometry = FunctionSpace(topology, VectorElement(P1_TRIANGLE, 2))
    coords = Field(geometry, VERTS, name="positions")
    mesh = Mesh(topology, coords)
    space = FunctionSpace(mesh.topology, VectorElement(P1_TRIANGLE, 2))
    u = TrainableField(
        space,
        0.1 * torch.ones(4, 2),
        bcs=[
            DirichletBC(0, BOTTOM, value=[0.0, 0.0]),
            DirichletBC(0, TOP, value=[0.0, -1.0]),
        ],
        name="u",
    )
    assert u.values_reduced.numel() == 0
