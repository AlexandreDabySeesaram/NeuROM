import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import library modules
from neurom.quadratures import TwoPoints1D
from neurom.meshes import Topology, Mesh
from neurom.elements import VectorElement, P1_BAR
from neurom.function_space import FunctionSpace, DirichletBC
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    PointWiseInterpolator,
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)

from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


def main():
    N = 40
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    # Positions
    x_min = 0.0
    x_max = 6.28
    x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)

    # Initialize displacement value
    u_init = 0.5 * torch.ones(N, 1)

    # Define constant load
    load = 1000.0 * torch.ones(N, 1)

    # Topology shared by the (trainable) geometry and the fields
    topology = Topology(elements)

    # Define quadrature method
    quad = TwoPoints1D()

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    # Positions: trainable (moving mesh / r-adaptivity), end vertices pinned to the
    # domain bounds. The geometry is a P1 vector field (1 component = the x coord).
    geometry_space = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    x = field_layout.add(
        TrainableField(
            geometry_space,
            x_array,
            bcs=[
                DirichletBC(0, [0], value=x_min),
                DirichletBC(0, [N - 1], value=x_max),
            ],
            name="positions",
        ),
    )

    # Generate mesh from the topology and the trainable coordinate field
    mesh = Mesh(topology, x)

    # Function space the solution fields live on
    space = FunctionSpace(mesh.topology, P1_BAR)

    # Displacement
    u = field_layout.add(
        TrainableField(
            space,
            u_init,
            bcs=[DirichletBC(0, [0, N - 1], value=0.0)],
            name="displacement",
        )
    )

    # Load
    f = field_layout.add(Field(space, load, name="load"))

    # Define interpolation at quadrature points.
    # The positions are trainable (moving mesh / r-adaptivity), so the context
    # geometry must be recomputed at each step via domain.update_contexts().
    ctx = QuadratureContext(mesh, quad)
    assembly_u = QuadratureAssembly(ctx, u)
    assembly_f = QuadratureAssembly(ctx, f)
    domain = IntegrationDomain([assembly_u, assembly_f])

    # Define physics to solve
    mu = 1e12
    physics = ElasticEnergy(field=u) + LoadPotential(field=u, f=f)

    class FlipLoss(nn.Module):
        def __init__(self, x, mu: float):
            super().__init__()
            self.x = x
            self.mu = mu

        def forward(self) -> float:
            x_nodes = self.x.at_elements()
            J = 0.5 * (x_nodes[:, 1, :] - x_nodes[:, 0, :])
            result = self.mu * torch.sum(torch.relu(-J) ** 2)
            return result

    class TotalLoss(nn.Module):
        def __init__(self, losses):
            super().__init__()
            self.losses = losses

        def forward(self) -> float:
            result = sum([loss_term.forward() for loss_term in self.losses])
            return result

    # Potential energy part of the loss
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    flip_loss = FlipLoss(x=x, mu=mu)

    total_loss = TotalLoss([physics_loss, flip_loss])

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=total_loss,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    loss_history = []

    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 6000
    for i in range(n_epochs):
        # Positions are trainable: recompute the geometry before interpolation.
        domain.update_contexts()
        loss = model()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    # Freeze position
    x.freeze()

    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=physics_loss,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    print("* Second training")
    for i in range(n_epochs):
        domain.update_contexts()
        loss = model()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    print("\n* Evaluation")
    # At quadrature points
    result = field_layout["displacement"]

    # At test points (shape (N, 1): at_position wants one point per row, not a flat list)
    x_test = torch.linspace(0, 6, 30).unsqueeze(-1)
    pwi = PointWiseInterpolator(mesh, u)
    u_test = pwi.at_position(x_test).squeeze().detach()
    if plot_loss:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    if plot_test:
        plt.figure()
        plt.plot(
            result.x.values.flatten().detach(),
            result.u.values.flatten().detach(),
            "+",
            label="Gauss points",
        )
        plt.plot(x_test, u_test, "o", label="Test points")
        plt.xlabel("x [mm]")
        plt.ylabel("u(x) [mm]")
        plt.title("Displacement Field")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
