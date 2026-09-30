import numpy as np
import torch


# Import library modules
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import HermiteBeam, LinearBar
from neurom.meshes import Connectivity
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
    PointWiseInterpolator,
)
from neurom.physics import SolidElasticEnergy, MembraneStretchEnergy
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel
from neurom.math import jacobian, second_derivative


torch.set_default_dtype(torch.float64)


# Initial DOFs.
#########################################################


def symmetric_mode(eps, m=1):
    """Clamped-clamped symmetric mode w = eps (1 - cos 2 m pi x).

    Returns:
        callable: x -> w, of the shape of x.
    """
    k = 2 * m * torch.pi
    return lambda x: eps * (1 - torch.cos(k * x))


def antisymmetric_gamma(n=1):
    """n-th positive root of tan(gamma/2) = gamma/2, by Newton from (2n+1) pi.

    Written as h(t) = sin t - t cos t = 0 with t = gamma/2 (no tan poles).
    """
    t = (2 * n + 1) * torch.pi / 2 - 1e-2
    for _ in range(50):
        t = t - (np.sin(t) - t * np.cos(t)) / (t * np.sin(t))
    return 2 * t


def antisymmetric_mode(eps, n=1):
    """Clamped-clamped n-th antisymmetric mode.

    w = eps (1 - 2x - cos(gamma x) + 2/gamma sin(gamma x)),
    gamma = antisymmetric_gamma(n).

    Returns:
        callable: x -> w, of the shape of x.
    """
    g = antisymmetric_gamma(n)
    return lambda x: eps * (1 - 2 * x - torch.cos(g * x) + 2 / g * torch.sin(g * x))


def nodal_dofs(w_init, x):
    """Hermite DOFs (w, w') at positions x, w' obtained by autograd.

    Args:
        w_init (callable): x -> w, any differentiable torch function.
        x (torch.Tensor): positions, shape (N_nodes,).

    Returns:
        tuple: w and w', shape (N_nodes,), detached.
    """
    x = x.detach().clone().requires_grad_(True)
    w = w_init(x)
    (dw,) = torch.autograd.grad(w.sum(), x)
    return w.detach(), dw.detach()


# Problem.
#########################################################


def build_problem(n_elements, P, w_init):
    """Assemble the clamped-clamped von Karman beam on [0, 1].

    Args:
        n_elements (int): number of Hermite elements.
        P (float): non-dimensional axial load.
        w_init (callable): x -> w, initial deflection; the slope DOFs w'
            are obtained by autograd (see `nodal_dofs`).

    Returns:
        dict: model, field_layout, domain, the displacement field `u`, its
        shape function and mapping, the `stretch` term, the node positions and P.
    """
    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    ## Mesh
    N = n_elements + 1
    x_array = torch.linspace(0.0, 1.0, N).unsqueeze(-1)

    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T # [e, e+1] for e in range(N-1)

    connectivity_geom = Connectivity(nodes, elements)

    # Positions
    x = field_layout.add(
        Field(name="positions", connectivity=connectivity_geom, values=x_array)
    )

    # Generate mesh
    mesh = Mesh(connectivity=connectivity_geom, nodes_positions=x)

    # Define mapping (for positions only)
    sf_geom = LinearBar()
    mapping_geom = IsoparametricMapping1D(sf_geom, mesh)

    ## Field
    # 2 DOFs per node (w and its physical slope w')
    x_n = x_array.squeeze(-1)
    u_init = torch.empty(2 * N, 1)
    u_init[0::2, 0], u_init[1::2, 0] = nodal_dofs(w_init, x_n)

    nodes = torch.arange(0, 2 * N)
    elements = torch.vstack([
        torch.arange(0, 2*N - 2, 2),   # 0, 2, ..., 2N-4
        torch.arange(1, 2*N - 1, 2),   # 1, 3, ..., 2N-3
        torch.arange(2, 2*N,     2),   # 2, 4, ..., 2N-2
        torch.arange(3, 2*N + 1, 2),   # 3, 5, ..., 2N-1
    ]).T

    connectivity_field = Connectivity(nodes, elements)

    # Boundary conditions (clamped: w = w' = 0 at both ends)
    nodes_u_bc = [0, 1, 2 * N - 2, 2 * N - 1]
    u_bc = torch.zeros(4, 1)

    # Displacement
    u = field_layout.add(
        TrainableField(
            name="displacement",
            connectivity=connectivity_field,
            init_values=u_init,
            constraint=Dirichlet(nodes=nodes_u_bc, values_imposed=u_bc),
        )
    )

    sf_field = HermiteBeam()

    # Quadrature strategy: two Gauss points per element.
    quad = TwoPoints1D()

    # Define interpolation at quadrature
    ctx = QuadratureContext(mesh, quad, mapping_geom)
    assembly_u = QuadratureAssembly(ctx, sf_field, u)
    domain = IntegrationDomain([assembly_u])

    ## Energy
    # Bending
    bending_energy = SolidElasticEnergy(u, strain=second_derivative, stress_point=lambda kappa: kappa)

    # Axial load potential: -P/2 * int w'^2
    axial_loading_energy = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: -P * eps)

    # Stretch: integrand w'^2 dx (SolidElasticEnergy has a built-in 1/2, hence 2 * eps)
    stretch = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: 2 * eps)

    # Membrane stretching: 1/8 * (int w'^2)^2
    membrane_stretching_energy = MembraneStretchEnergy(stretch)

    energy = bending_energy + axial_loading_energy + membrane_stretching_energy
    loss = PhysicsLoss(energy, field_layout)

    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=loss,
    )

    return {
        "model": model,
        "field_layout": field_layout,
        "domain": domain,
        "mesh": mesh,
        "mapping": mapping_geom,
        "sf": sf_field,
        "u": u,
        "stretch": stretch,
        "x_nodes": x_n,
        "P": P,
    }


# Minimization.
#########################################################


def minimize(model, *, adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol, verbose=True):
    """Minimize the model energy with Adam, then LBFGS (strong Wolfe).

    Each phase stops after its number of iterations or when the relative change
    of the loss |L_k - L_{k-1}| / |L_k| falls below `tol`. A phase with 0
    iterations is skipped. One LBFGS iteration is one `optimizer.step`.

    Returns:
        dict: "loss" (list of floats) and "phase" (list of "adam" / "lbfgs").
    """
    history = {"loss": [], "phase": []}

    phases = [
        ("adam", adam_iters, lambda: torch.optim.Adam(model.parameters(), lr=adam_lr)),
        ("lbfgs", lbfgs_iters, lambda: torch.optim.LBFGS(
            model.parameters(), lr=lbfgs_lr, line_search_fn="strong_wolfe")),
    ]

    for name, n_iters, make_optimizer in phases:
        if n_iters == 0:
            continue
        optimizer = make_optimizer()

        def closure():
            optimizer.zero_grad()
            loss = model()
            if loss.requires_grad:
                loss.backward()
            return loss

        previous = None
        for i in range(n_iters):
            loss = optimizer.step(closure).item()
            history["loss"].append(loss)
            history["phase"].append(name)
            if verbose:
                print(f"{name} i = {i} loss={loss:.6e}")
            if previous is not None and abs(loss - previous) <= tol * max(abs(loss), 1e-30):
                break
            previous = loss

    return history


# Post-processing.
#########################################################


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
    I_e = stretch.integrand(field_layout).detach().reshape(x_nodes.shape[0] - 1, -1).sum(dim=1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(I_e, dim=0)])
    S = cum[-1]

    return -((P - 0.5 * S) * x_nodes + 0.5 * cum)


def sample_deflection(problem, n_per_element=20):
    """Evaluate w with the Hermite shape functions on a fine grid.

    Returns:
        tuple: x (N_pts,) and w (N_pts,), non-dimensional.
    """
    n_elements = problem["x_nodes"].shape[0] - 1
    x = torch.linspace(0.0, 1.0, n_elements * n_per_element + 1)
    pwi = PointWiseInterpolator(problem["mesh"], problem["sf"], problem["u"], problem["mapping"])
    w = pwi.at_position(x.unsqueeze(-1)).squeeze(-1).detach()
    return x, w


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


def main(
    *, n_elements, P, w_init, adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol,
    E, A, I, length,
):
    """Build, minimize and post-process the beam problem.

    Args:
        n_elements (int): number of Hermite elements.
        P (float): non-dimensional axial load.
        w_init (callable): x -> w, initial deflection (slopes by autograd).
        adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol: see `minimize`.
        E, A, I, length (float): physical constants, used by `rescale`.

    Returns:
        dict: dimensional and non-dimensional results, loss and phase history.
    """
    problem = build_problem(n_elements, P, w_init)

    print("* Training")
    history = minimize(
        problem["model"],
        adam_lr=adam_lr,
        adam_iters=adam_iters,
        lbfgs_lr=lbfgs_lr,
        lbfgs_iters=lbfgs_iters,
        tol=tol,
    )

    # Refresh quadrature results with the trained DOFs
    field_layout = problem["field_layout"]
    problem["domain"].interpolate_all(field_layout)

    x_nodes = problem["x_nodes"]
    w_nodes = problem["u"].full_values()[0::2].squeeze(-1).detach()
    u_axial = axial_displacement(problem["stretch"], field_layout, x_nodes, P)

    x_hat, w_hat, u_hat, P_hat = rescale(
        x_nodes, w_nodes, u_axial, P, E=E, A=A, I=I, length=length
    )

    # Fine Hermite sampling of w, axial displacement linearly interpolated
    x_fine, w_fine = sample_deflection(problem)
    u_fine = torch.from_numpy(
        np.interp(x_fine.numpy(), x_nodes.numpy(), u_axial.numpy())
    ).to(x_fine.dtype)
    x_fine_hat, w_fine_hat, u_fine_hat, _ = rescale(
        x_fine, w_fine, u_fine, P, E=E, A=A, I=I, length=length
    )

    return {
        "x_hat": x_hat,
        "w_hat": w_hat,
        "u_hat": u_hat,
        "P_hat": P_hat,
        "x_fine_hat": x_fine_hat,
        "w_fine_hat": w_fine_hat,
        "u_fine_hat": u_fine_hat,
        "x": x_nodes,
        "w": w_nodes,
        "u_axial": u_axial,
        "P": P,
        "loss_history": history["loss"],
        "phase_history": history["phase"],
    }


if __name__ == "__main__":
    from plotting import plot_deformed_beam

    # Physical constants (steel, 1 cm x 1 cm square section)
    E = 210e9  # Young's modulus [Pa]
    b = 1e-2  # section side [m]
    A = b**2  # cross-section area [m^2]
    I = b**4 / 12  # second moment of area [m^4]
    L = 1.0  # beam length [m]

    # Numerical parameters
    N_ELEMENTS = 10
    P = 1.5 * 4 * torch.pi**2  # non-dimensional axial load (first critical load 4 pi^2)
    # Initial deflection x -> w (slopes by autograd). Other choices:
    # antisymmetric_mode(1e-2, n=1), symmetric_mode(1e-2, m=2), or any
    # differentiable lambda x: ... vanishing with its slope at x = 0 and 1.
    W_INIT = symmetric_mode(eps = 1e-2, m=1)

    # Minimization hyperparameters
    ADAM_LR = 1e-1
    ADAM_ITERS = 500
    LBFGS_LR = 1.0
    LBFGS_ITERS = 0
    TOL = 1e-6

    results = main(
        n_elements=N_ELEMENTS, P=P, w_init=W_INIT,
        adam_lr=ADAM_LR, adam_iters=ADAM_ITERS,
        lbfgs_lr=LBFGS_LR, lbfgs_iters=LBFGS_ITERS, tol=TOL,
        E=E, A=A, I=I, length=L,
    )


    plot_deformed_beam(
        results["x_fine_hat"], results["w_fine_hat"], results["u_fine_hat"],
        x_nodes=results["x_hat"], w_nodes=results["w_hat"], u_nodes=results["u_hat"],
        amplification=5.0,
    ).show()
