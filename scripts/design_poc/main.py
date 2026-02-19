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
from neurom.field import Field
from neurom.integrator import Integrator
from neurom.interpolator import Interpolator
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
    x_array = torch.linspace(x_min, x_max, N)[:, None]
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    topology = Topology(nodes, elements)

    sf = LinearSegment()
    quad = MidPoint1D()
    # quad = quadratures.TwoPoints1D()
    mapping = IsoparametricMapping1D(sf)
    # Unknown
    u_init = 0.5 * torch.ones(N, 1)
    u = Field(
        name="displacement",
        topology=topology,
        init_values=u_init,
        constraint=Dirichlet([0, N - 1]),
        trainable=True,
    )
    # Positions
    x = Field(
        name="positions",
        topology=topology,
        init_values=x_array,
        constraint=Dirichlet(dirichlet_nodes=[0, N - 1], values_imposed=[x_min, x_max]),
        trainable=True,
    )
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
    n_epochs = 70000
    for i in range(n_epochs):
        loss = model()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    print("\n* Evaluation")
    # At quadrature points
    x_q, u_q, _ = model.interpolator.interpolate()

    # At test points
    x_test = torch.linspace(0, 6, 30)
    u_test = model.interpolator.interpolate_at(x_test).squeeze()
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
            x_q.flatten().detach(), u_q.flatten().detach(), "+", label="Gauss points"
        )
        plt.plot(x_test, u_test, "o", label="Test points")
        plt.xlabel("x [mm]")
        plt.ylabel("u(x) [mm]")
        plt.title("Displacement Field")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
