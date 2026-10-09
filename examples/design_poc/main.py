import torch
import matplotlib.pyplot as plt

# Import library modules
from neurom.quadratures import TwoPoints1D
from neurom.meshes import Mesh, Topology
from neurom.elements import P1_BAR, VectorElement
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

    # Mesh (topology + P1 geometry) and the function space the fields live on
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, x_array, name="positions")
    mesh = Mesh(topology, coords)
    space = FunctionSpace(mesh.topology, P1_BAR)

    # Define quadrature method
    quad = TwoPoints1D()

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

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

    # Define physics to solve
    physics = ElasticEnergy(field=u) + LoadPotential(field=u, f=f)

    # Potential energy part of the loss
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    # Define quadrature context
    ctx = QuadratureContext(mesh, quad)

    # Define quadrature assemblies
    assembly_u = QuadratureAssembly(ctx, u)
    assembly_f = QuadratureAssembly(ctx, f)

    domain = IntegrationDomain([assembly_u, assembly_f])

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=physics_loss,
    )

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1e-1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(retain_graph=True)
        return loss

    loss_history = []

    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 5
    for i in range(n_epochs):
        loss = model()
        optimizer.step(closure)

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
