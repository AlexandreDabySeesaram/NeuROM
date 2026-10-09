import torch


# Import library modules
from neurom.quadratures import TwoPoints1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.elements import HERMITE, P1_BAR, VectorElement
from neurom.function_space import FunctionSpace, DirichletBC
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
    interpolate,
)
from neurom.physics import SolidElasticEnergy, MembraneStretchEnergy
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel
from neurom.math import jacobian, second_derivative


# Physical constants (steel, 1 cm x 1 cm square section)
E = 210e9  # Young's modulus [Pa]
b = 1e-2  # section side [m]
A = b**2  # cross-section area [m^2]
I = b**4 / 12  # second moment of area [m^4]
L = 1.0  # beam length [m]


def main():

    # Mesh, shape functions, quadrature, mapping, boundary conditions.
    ##################################################################

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    ## Mesh

    # Dimensions
    x_min = 0.0
    x_max = 1.0
    N = 5

    # Generate vertices and connectivity
    x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)

    elements = torch.vstack(
        [torch.arange(0, N - 1), torch.arange(1, N)]
    ).T  # [e, e+1] for e in range(N-1)

    # Generate mesh (topology + P1 geometry)
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    coords = Field(geometry, x_array, name="positions")
    mesh = Mesh(topology, coords)

    ## Field space: cubic Hermite. The element owns the DOF layout (per vertex: a
    ## value and a slope); the FunctionSpace deduces the numbering and connectivity,
    ## so nothing here mentions "2 DOFs per node" or raw DOF indices. The space knows
    ## nothing of quadrature -- that is supplied at assembly time.
    space = FunctionSpace(mesh.topology, HERMITE)

    # Seed the field from a shape w0 = eps (1 - cos 2 pi x). Interpolation is a *separate*,
    # optional step: `interpolate` builds the FE interpolant's DOF values (value DOFs <- w0,
    # slope DOFs <- w0' via autograd); the field itself just takes those raw values -- its
    # init is a seed for training, not tied to any DOF's "meaning". Clamped-clamped: fix the
    # value and slope at both end vertices (by meaning).
    eps_init = 1e-2
    u0 = interpolate(
        space,
        mesh,
        lambda z: (eps_init * (1 - torch.cos(2 * torch.pi * z))).squeeze(-1),
    )
    u = field_layout.add(
        TrainableField(
            space,
            u0,
            bcs=[DirichletBC(0, [0, N - 1], kinds=["value", "d1"])],
            name="displacement",
        )
    )

    # Interpolation at quadrature points: quadrature + mapping enter here, at the
    # integration layer, not in the space.
    quad = TwoPoints1D()
    ctx = QuadratureContext(mesh, quad)
    assembly_u = QuadratureAssembly(ctx, u)
    domain = IntegrationDomain([assembly_u])

    # Energy.
    #########################################################

    # Bending
    bending_energy = SolidElasticEnergy(
        u, strain=second_derivative, stress_point=lambda kappa: kappa
    )

    # Axial load (non-dimensional P, first clamped-clamped critical load is 4*pi^2)
    P = 1.5 * 4 * torch.pi**2

    # Axial load potential: -P/2 * int w'^2
    axial_loading_energy = SolidElasticEnergy(
        u, strain=jacobian, stress_point=lambda eps: -P * eps
    )

    # Membrane stretching: 1/8 * (int w'^2)^2

    # Stretch: integrand w'^2 dx (SolidElasticEnergy has a built-in 1/2, hence 2 * eps)
    stretch = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: 2 * eps)

    membrane_stretching_energy = MembraneStretchEnergy(stretch)

    # Sum the pieces
    energy = bending_energy + axial_loading_energy + membrane_stretching_energy
    # Define loss
    loss = PhysicsLoss(energy, field_layout)

    # FEM Model.
    #########################################################

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=loss,
    )

    # Optimizer.
    #########################################################

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-1,  # line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = model()
        if loss.requires_grad:
            loss.backward()
        return loss

    loss_history = []

    print("* Training")
    n_epochs = 1000
    for i in range(n_epochs):
        print(i)
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        print(
            f"i = {i} loss={loss.item():.3e}",
        )

    # Post-processing.
    #########################################################

    # Refresh quadrature results with the trained DOFs
    domain.interpolate_all(field_layout)

    x_nodes = x_array.squeeze(-1)
    w_nodes = (
        u.full_values()[space.locate(0, range(N), ["value"])].squeeze(-1).detach()
    )  # value DOFs, by meaning
    u_axial = axial_displacement(stretch, field_layout, x_nodes, P)

    x_hat, w_hat, u_hat, P_hat = rescale(
        x_nodes, w_nodes, u_axial, P, E=E, A=A, I=I, length=L
    )

    return {
        "x_hat": x_hat,
        "w_hat": w_hat,
        "u_hat": u_hat,
        "P_hat": P_hat,
        "x": x_nodes,
        "w": w_nodes,
        "u_axial": u_axial,
        "P": P,
        "loss_history": loss_history,
    }


def axial_displacement(stretch, field_layout, x_nodes, P):
    """Non-dimensional axial displacement at the mesh nodes.

    u(x_j) = -[(P - S/2) x_j + 1/2 int_0^{x_j} w'^2], with S = int_0^1 w'^2,
    so that u_hat = (r^2 / l) * u. The integrals reuse the quadrature of
    `stretch` (integrand w'^2 dx), i.e. the one the solver minimized.

    Args:
        stretch (SolidElasticEnergy): term with integrand w'^2 dx.
        field_layout (FieldLayout): layout holding the up-to-date quadrature results.
        x_nodes (torch.Tensor): non-dimensional node positions, shape (N_nodes,).
        P (float): non-dimensional axial load.

    Returns:
        torch.Tensor: non-dimensional axial displacement, shape (N_nodes,).
    """
    # Per-element integrals of w'^2, then cumulative from x = 0
    I_e = (
        stretch.integrand(field_layout)
        .detach()
        .reshape(x_nodes.shape[0] - 1, -1)
        .sum(dim=1)
    )
    cum = torch.cat([torch.zeros(1), torch.cumsum(I_e, dim=0)])
    S = cum[-1]

    return -((P - 0.5 * S) * x_nodes + 0.5 * cum)


def rescale(x, w, u_axial, P, *, E, A, I, length):
    """Convert non-dimensional quantities back to physical ones.

    x_hat = l x, w_hat = r w, u_hat = (r^2 / l) u, P_hat = P E I / l^2,
    with r = sqrt(I / A).
    """
    r = (I / A) ** 0.5
    return (
        length * x,
        r * w,
        (r**2 / length) * u_axial,
        P * E * I / length**2,
    )


if __name__ == "__main__":
    main()
