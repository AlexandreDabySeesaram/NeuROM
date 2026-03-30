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
from neurom.field_layout import FieldLayout
from neurom.interpolation import PointWiseInterpolator, Interpolator, FieldInterpolator

from neurom.physics import ElasticEnergy, LoadPotential, BarrierPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


def main():
    N = 40
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    # Positions
    x_min = 0.0
    x_max = 6.28
    x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)
    h_0 = x_array[1] - x_array[0]
    # Initialize displacement value
    u_init = 0.5 * torch.ones(N, 1)

    # Define constant load
    load = 1000.0 * torch.ones(N, 1)

    # Initialize topology
    topology = Topology(nodes, elements)

    # Define shape function to use
    sf = LinearSegment()
    # Define quadrature method
    quad = TwoPoints1D()
    # Define mapping (for positions only)
    mapping = IsoparametricMapping1D(sf)

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    # Displacement
    u = field_layout.add(
        TrainableField(
            name="displacement",
            topology=topology,
            init_values=u_init,
            constraint=Dirichlet(nodes=[0, N - 1], values_imposed=torch.zeros(2, 1)),
        )
    )

    # Positions
    # x = field_layout.add(Field(name="positions", topology=topology, values=x_array))

    # Positions
    x = field_layout.add(
        TrainableField(
            name="positions",
            topology=topology,
            init_values=x_array,
            constraint=Dirichlet(
                nodes=[0, N - 1],
                values_imposed=torch.tensor([x_min, x_max]).unsqueeze(-1),
            ),
        ),
    )

    # Load
    f = field_layout.add(Field(name="load", topology=topology, values=load))

    # Generate mesh
    mesh = Mesh(topology=topology, nodes_positions=x)

    # Define interpolator
    interpolator = Interpolator(
        mesh,
        quad,
        mapping,
        [FieldInterpolator(sf, u), FieldInterpolator(sf, f)],
    )

    # Define physics to solve
    alpha = 1e6
    mu = 1e12
    physics = ElasticEnergy(field=u) + LoadPotential(field=u, f=f)

    class RegularizationLoss(nn.Module):
        def __init__(self, x, alpha: float, h_0: float):
            super().__init__()
            self.x = x
            self.h_0 = h_0
            self.alpha = alpha

        def forward(self) -> float:
            x_nodes = self.x.at_elements()
            result = self.alpha * torch.mean(
                (x_nodes[:, 1, :] - x_nodes[:, 0, :] - self.h_0) ** 2
            )
            return result

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
            result = sum([l.forward() for l in self.losses])
            return result

    # Potential energy part of the loss
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    flip_loss = FlipLoss(x=x, mu=mu)

    total_loss = TotalLoss([physics_loss, flip_loss])

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        interpolator=interpolator,
        loss=total_loss,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    loss_history = []

    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 6000
    for i in range(n_epochs):
        loss = model()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")
        # print(f"x = {x.full_values()}")

    # Freeze position
    x.freeze()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    print("* Second training")
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
    result = field_layout["displacement"]

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
