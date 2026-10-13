import numpy as np
import torch


# Import library modules
from neurom.quadratures import TwoPoints1D, ThreePoints1D
from neurom.shape_functions import CubicHermiteBar, LinearBar
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
)
from neurom.physics import SolidElasticEnergy, MembraneStretchEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel
from neurom.math import jacobian, hessian

from modes import symmetric_mode, nodal_dofs
from utils import minimize, axial_displacement, sample_deflection, rescale


torch.set_default_dtype(torch.float64)


# Problem.
#########################################################


def build_problem(
    n_elements, P, w_init=None, *, k=0.0, load=None, noise=0.0, seed=None, quadrature=None,
):
    """Assemble the clamped-clamped von Karman beam on [0, 1].

    Pi = 1/2 int w''^2 - P/2 int w'^2 + 1/8 (int w'^2)^2 + k/2 int w^2 - int f w.
    The analytical solutions of `modes` hold for k = 0 and f = 0 only.

    Args:
        n_elements (int): number of Hermite elements.
        P (float): non-dimensional axial load.
        w_init (callable, optional): x -> w, initial deflection; the slope DOFs
            w' are obtained by autograd (see `nodal_dofs`). None means w = 0.
        k (float): stiffness of the elastic (Winkler) foundation.
        load (callable, optional): x -> f, transverse load density, interpolated
            with the Hermite shape functions (slopes by autograd). None means f = 0.
        noise (float): amplitude of a Gaussian noise added to all DOFs (w, w').
        seed (int, optional): `torch.manual_seed` used for the noise.
        quadrature (QuadratureRule, optional): defaults to TwoPoints1D().

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
    # We keep it 1D but with twice the number of nodes. Stride of 2 because the
    # DOFs are stored intertwined (w, w'), to match the ordering of CubicHermiteBar.
    u_init = torch.zeros(2 * N, 1)
    if w_init is not None:
        u_init[0::2, 0], u_init[1::2, 0] = nodal_dofs(w_init, x_n)
    if noise:
        if seed is not None:
            torch.manual_seed(seed)
        u_init += noise * torch.randn(2 * N, 1)

    nodes = torch.arange(0, 2 * N)

    # elements[i] gives the list of the indices of the DoFs inside element i
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

    # Transverse load density, same Hermite DOFs (f, f') as the displacement
    f_values = torch.zeros(2 * N, 1)
    if load is not None:
        f_values[0::2, 0], f_values[1::2, 0] = nodal_dofs(load, x_n)
    f = field_layout.add(
        Field(name="load", connectivity=connectivity_field, values=f_values)
    )

    # Shape function for the interpolation
    sf_field = CubicHermiteBar()

    # Quadrature strategy (default: two Gauss points per element).
    quad = TwoPoints1D() if quadrature is None else quadrature

    # Define interpolation at quadrature
    ## Quad context: meqsure and points of quadratures
    ctx = QuadratureContext(mesh, quad, mapping_geom)
    ## Quad assembly, ready to be called 
    assembly_u = QuadratureAssembly(ctx, sf_field, u)
    assembly_f = QuadratureAssembly(ctx, sf_field, f)
    domain = IntegrationDomain([assembly_u, assembly_f])

    ## Energy
    # Bending
    bending_energy = SolidElasticEnergy(u, strain=hessian, stress_point=lambda kappa: kappa)

    # Axial load potential: -P/2 * int w'^2
    axial_loading_energy = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: -P * eps)

    # Stretch: integrand w'^2 dx (SolidElasticEnergy has a built-in 1/2, hence 2 * eps)
    stretch = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: 2 * eps)

    # Membrane stretching: 1/8 * (int w'^2)^2
    membrane_stretching_energy = MembraneStretchEnergy(u, coefficient=1 / 8)

    # Elastic foundation: k/2 * int w^2
    foundation_energy = SolidElasticEnergy(u, strain=lambda x, w: w, stress_point=lambda w: k * w)

    # Transverse load potential: -int f w
    load_potential = LoadPotential(u, f)

    energy = (
        bending_energy + axial_loading_energy + membrane_stretching_energy
        + foundation_energy + load_potential
    )
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


def main(
    *, n_elements, P, w_init, adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol,
    E, A, I, length, k=0.0, load=None, quadrature=None,
):
    """Build, minimize and post-process the beam problem.

    Args:
        n_elements (int): number of Hermite elements.
        P (float): non-dimensional axial load.
        w_init (callable): x -> w, initial deflection (slopes by autograd).
        adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol: see `minimize`.
        E, A, I, length (float): physical constants, used by `rescale`.
        k (float), load (callable, optional): see `build_problem`.
        quadrature (QuadratureRule, optional): see `build_problem`.

    Returns:
        dict: dimensional and non-dimensional results, loss and phase history.
    """
    problem = build_problem(n_elements, P, w_init, k=k, load=load, quadrature=quadrature)

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
    LBFGS_ITERS = 10
    TOL = 1e-6

    results = main(
        n_elements=N_ELEMENTS, P=P, w_init=W_INIT,
        adam_lr=ADAM_LR, adam_iters=ADAM_ITERS,
        lbfgs_lr=LBFGS_LR, lbfgs_iters=LBFGS_ITERS, tol=TOL,
        E=E, A=A, I=I, length=L, quadrature=ThreePoints1D(),
    )


    plot_deformed_beam(
        results["x_fine_hat"], results["w_fine_hat"], results["u_fine_hat"],
        x_nodes=results["x_hat"], w_nodes=results["w_hat"], u_nodes=results["u_hat"],
        amplification=5.0,
    ).show()
