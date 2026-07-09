import torch

from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.constraints import Dirichlet
from neurom.field_layout import FieldLayout
from neurom.interpolation import QuadratureContext, QuadratureAssembly, IntegrationDomain
from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


def test_forward_refresh_lets_gradients_reach_node_positions():
    n = 5
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    top = Topology(nodes, elements)

    sf = LinearSegment()
    quad = TwoPoints1D()
    mapping = IsoparametricMapping1D(sf)

    layout = FieldLayout()
    u = layout.add(
        TrainableField(
            name="u",
            topology=top,
            init_values=0.5 * torch.ones(n, 1),
            constraint=Dirichlet(nodes=[0, n - 1], values_imposed=torch.zeros(2, 1)),
        )
    )
    f = layout.add(Field(name="load", topology=top, values=torch.ones(n, 1)))

    mesh = Mesh.with_trainable_positions_1d(top, x)

    physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)
    loss = PhysicsLoss(physics=physics, field_layout=layout)

    ctx = QuadratureContext(mesh, quad, mapping)
    domain = IntegrationDomain(
        [QuadratureAssembly(ctx, sf, u), QuadratureAssembly(ctx, sf, f)]
    )
    model = FEMModel(
        mesh=mesh, field_layout=layout, integration_domain=domain, loss=loss
    )

    assert model.mesh.has_trainable_positions

    out = model()
    out.backward()

    coords = mesh.nodes_positions.coordinates
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0
