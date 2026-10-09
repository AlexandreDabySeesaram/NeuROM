import pytest
import torch

from neurom.fem_model import FEMModel
from neurom.field_layout import FieldLayout
from neurom.fields import Field, TrainableField
from neurom.function_space import FunctionSpace, DirichletBC
from neurom.interpolation import (
    IntegrationDomain,
    QuadratureAssembly,
    QuadratureContext,
)
from neurom.math import jacobian
from neurom.meshes import Mesh, Topology
from neurom.physics import SolidElasticEnergy
from neurom.physics_loss import PhysicsLoss
from neurom.quadratures import TwoPoints1D
from neurom.elements import P1_BAR, HERMITE, VectorElement


def build(element, n_dofs_per_node, strain, N=5):
    """1D model on [0, 1] with energy ½ ∫ strain(x, u)^2, using the finite ``element``."""
    fl = FieldLayout()
    elements = torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, points)
    mesh = Mesh(topology, coords)
    ctx = QuadratureContext(mesh, TwoPoints1D())

    d = n_dofs_per_node
    space = FunctionSpace(mesh.topology, element)
    # Original Dirichlet(nodes=[0]) fixed only global DOF 0 -- the value DOF at
    # vertex 0 -- leaving a Hermite slope DOF free; restrict the BC to the value
    # kind so Hermite's derivative DOF at vertex 0 stays free.
    u = fl.add(
        TrainableField(
            space,
            torch.rand(d * N, 1),
            bcs=[DirichletBC(0, [0], kinds=["value"], value=0.0)],
            name="u",
        )
    )
    domain = IntegrationDomain([QuadratureAssembly(ctx, u)])
    loss = PhysicsLoss(
        SolidElasticEnergy(u, strain=strain, stress_point=lambda e: e), fl
    )
    return FEMModel(
        mesh=mesh, field_layout=fl, integration_domain=domain, loss=loss
    ), ctx


def values(x, u):
    return u


@pytest.mark.parametrize(
    "element, d, strain, consumed",
    [
        (
            P1_BAR,
            1,
            jacobian,
            False,
        ),  # u' of linear element: loss independent of xi_back
        (P1_BAR, 1, values, True),  # values of u depend on xi_back
        (HERMITE, 2, jacobian, True),  # u' of cubic element depends on xi_back
    ],
)
def test_repeated_backward(element, d, strain, consumed):
    model, ctx = build(element, d, strain)
    for _ in range(3):
        model().backward()
        assert ctx.graph_consumed is consumed
