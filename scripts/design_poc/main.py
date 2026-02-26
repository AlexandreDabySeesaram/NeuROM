from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import library modules
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology, Mesh
from neurom.constraints import NoConstraint, Dirichlet
from neurom.fields import Field, TrainableField
from neurom.integrator import Integrator
from neurom.interpolation import PointWiseInterpolator, Interpolator
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


# =========================================================
# Physics
# =========================================================
class PoissonPhysics:
    def __init__(self, f):
        self.f = f

    def integrand(self, x, u):
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        return 0.5 * du_dx**2 - self.f(x) * u


# =========================================================
# Force function
# =========================================================
def f(x):
    return 1000.0


# =========================================================
# Main
# =========================================================
def main():

    N = 40
    x_min = 0.0
    x_max = 6.28
    x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    topology = Topology(nodes, elements)

    sf = LinearSegment()
    quad = MidPoint1D()
    mapping = IsoparametricMapping1D(sf)
    # Unknown
    u_init = 0.5 * torch.ones(N, 1)
    u = TrainableField(
        name="displacement",
        topology=topology,
        init_values=u_init,
        constraint=Dirichlet(nodes=[0, N - 1], values_imposed=torch.zeros(2, 1)),
    )
    # Positions
    # x = TrainableField(
    #    name="positions",
    #    topology=topology,
    #    init_values=x_array,
    #    constraint=Dirichlet(
    #        nodes=[0, N - 1], values_imposed=torch.tensor([x_min, x_max]).unsqueeze(-1)
    #    ),
    # )
    x = Field(name="positions", topology=topology, values=x_array)
    # Generate mesh
    mesh = Mesh(topology=topology, nodes_positions=x)
    interpolator = Interpolator(mesh, u, sf, quad, mapping)
    physics = PoissonPhysics(f)
    integrator = Integrator()

    model = FEMModel(
        mesh=mesh,
        field=u,
        interpolator=interpolator,
        physics=physics,
        integrator=integrator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    loss_history = []

    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 2000
    for i in range(n_epochs):
        loss = model()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")
        # print(f"x = {x.full_values()}")

    print("\n* Evaluation")
    # At quadrature points
    result = model.interpolator.interpolate()

    # At test points
    x_test = torch.linspace(0, 6, 30)
    pwi = PointWiseInterpolator(mesh, sf, u, mapping)
    u_test = pwi.at_position(x_test).squeeze()
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
            result.x.flatten().detach(),
            result.u.flatten().detach(),
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
