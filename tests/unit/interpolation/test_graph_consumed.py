import pytest
import torch

from neurom.constraints import Dirichlet
from neurom.fem_model import FEMModel
from neurom.field_layout import FieldLayout
from neurom.fields import Field, TrainableField
from neurom.geometry import IsoparametricMapping1D
from neurom.interpolation import IntegrationDomain, QuadratureAssembly, QuadratureContext
from neurom.math import jacobian
from neurom.meshes import Connectivity, Mesh
from neurom.physics import SolidElasticEnergy
from neurom.physics_loss import PhysicsLoss
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import HermiteBeam, LinearBar


def build(sf, n_dofs_per_node, strain, N=5):
    """1D model on [0, 1] with energy ½ ∫ strain(x, u)^2, using the shape functions ``sf``."""
    fl = FieldLayout()
    geom = Connectivity(torch.arange(N), torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T)
    x = fl.add(Field(name="positions", connectivity=geom, values=torch.linspace(0, 1, N).unsqueeze(-1)))
    mesh = Mesh(connectivity=geom, nodes_positions=x)
    ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(LinearBar(), mesh))

    d = n_dofs_per_node
    elements = torch.stack([torch.arange(d * e, d * e + 2 * d) for e in range(N - 1)])
    u = fl.add(
        TrainableField(
            name="u",
            connectivity=Connectivity(torch.arange(d * N), elements),
            init_values=torch.rand(d * N, 1),
            constraint=Dirichlet(nodes=[0], values_imposed=torch.zeros(1, 1)),
        )
    )
    domain = IntegrationDomain([QuadratureAssembly(ctx, sf, u)])
    loss = PhysicsLoss(SolidElasticEnergy(u, strain=strain, stress_point=lambda e: e), fl)
    return FEMModel(mesh=mesh, field_layout=fl, integration_domain=domain, loss=loss), ctx


def values(x, u):
    return u


@pytest.mark.parametrize(
    "sf, d, strain, consumed",
    [
        (LinearBar(), 1, jacobian, False),  # u' of linear sf: loss independent of xi_back
        (LinearBar(), 1, values, True),  # values of u depend on xi_back
        (HermiteBeam(), 2, jacobian, True),  # u' of cubic sf depends on xi_back
    ],
)
def test_repeated_backward(sf, d, strain, consumed):
    model, ctx = build(sf, d, strain)
    for _ in range(3):
        model().backward()
        assert ctx.graph_consumed is consumed

