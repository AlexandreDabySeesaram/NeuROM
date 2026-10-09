import torch


# Import library modules
from neurom.quadratures import ThreePoints1D
from neurom.shape_functions import CubicHermiteBar, LinearBar
from neurom.meshes import Connectivity
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh
from neurom.constraints import Dirichlet, NoConstraint
from neurom.fields import Field
from neurom.field_layout import FieldLayout
from neurom.interpolation import IntegrationDomain, PointWiseInterpolator
from neurom.decompositions import CPPGD, FactorSpace, MonomSpec
from neurom.neurom_model import NeuROMModel
from neurom.training import GreedyTrainer, Phase, RelativeChange, MaxStages
from neurom.math import jacobian, hessian, inner, integrate

from modes import symmetric_mode, nodal_dofs


torch.set_default_dtype(torch.float64)


# Problem.
#########################################################


def build_problem_pgd(n_elements, n_P, P_range, w_init, *, quadrature=None):
    """Assemble the separated clamped-clamped von Karman beam w(x, P).

    w(x, P) = sum_m w_m(x) lambda_m(P), x in [0, 1], P in `P_range`.

    Args:
        n_elements (int): number of Hermite elements in x.
        n_P (int): number of linear elements in P.
        P_range (tuple): (P_min, P_max), non-dimensional axial loads.
        w_init (callable): x -> w, initial x monom; the slope DOFs w' are
            obtained by autograd (see `nodal_dofs`). The P monom starts at 1.
            Every new mode starts from the same values.
        quadrature (QuadratureRule, optional): used on both axes, defaults to
            ThreePoints1D() (exact for the quartic integrands of S and of the
            membrane term in P).

    Returns:
        dict: model, decomposition `cppgd`, field_layout, the node positions in
        x and the P range.
    """
    quad = ThreePoints1D() if quadrature is None else quadrature

    ## Space x
    N = n_elements + 1
    x_array = torch.linspace(0.0, 1.0, N).unsqueeze(-1)

    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T # [e, e+1] for e in range(N-1)

    connectivity_geom = Connectivity(nodes, elements)
    x = Field(name="x_positions", connectivity=connectivity_geom, values=x_array)
    mesh_x = Mesh(connectivity=connectivity_geom, nodes_positions=x)

    space_x = FactorSpace(
        name="x",
        mesh=mesh_x,
        mapping=IsoparametricMapping1D(LinearBar(), mesh_x),
        quad=quad,
    )

    # 2 DOFs per node (w and its physical slope w'), stored intertwined to
    # match the ordering of CubicHermiteBar
    x_n = x_array.squeeze(-1)
    w_dofs = torch.zeros(2 * N, 1)
    w_dofs[0::2, 0], w_dofs[1::2, 0] = nodal_dofs(w_init, x_n)

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

    spec_x = MonomSpec(
        space=space_x,
        sf=CubicHermiteBar(),
        constraint=Dirichlet(nodes=nodes_u_bc, values_imposed=u_bc),
        init_values=w_dofs,
        connectivity=connectivity_field,
    )

    ## Load P
    N_P = n_P + 1
    P_array = torch.linspace(P_range[0], P_range[1], N_P).unsqueeze(-1)

    nodes_P = torch.arange(0, N_P)
    elements_P = torch.vstack([torch.arange(0, N_P - 1), torch.arange(1, N_P)]).T

    connectivity_P = Connectivity(nodes_P, elements_P)
    P_positions = Field(name="P_positions", connectivity=connectivity_P, values=P_array)
    mesh_P = Mesh(connectivity=connectivity_P, nodes_positions=P_positions)

    space_P = FactorSpace(
        name="P",
        mesh=mesh_P,
        mapping=IsoparametricMapping1D(LinearBar(), mesh_P),
        quad=quad,
    )
    spec_P = MonomSpec(
        space=space_P,
        sf=LinearBar(),
        constraint=NoConstraint(),
        init_values=torch.ones(N_P, 1),
    )

    ## Decomposition and model
    cppgd = CPPGD(monom_specs=[spec_x, spec_P], name="pgd", n_modes_ini=1)

    # The domain holds no static assembly: the decomposition's own assemblies
    # are passed by the model at every forward.
    field_layout = FieldLayout()
    model = NeuROMModel(
        field_layout=field_layout,
        decomposition=cppgd,
        integration_domain=IntegrationDomain([]),
        loss=lambda out: energy(out, cppgd),
    )

    return {
        "model": model,
        "cppgd": cppgd,
        "field_layout": field_layout,
        "x_nodes": x_n,
        "P_range": P_range,
    }


# Energy.
#########################################################


def energy(field_layout, cppgd):
    """Separated von Karman energy, integrated over P.

    With K_mn = int w_m'' w_n'' dx, S_mn = int w_m' w_n' dx,
    A_mn = int lambda_m lambda_n dP, B_mn = int P lambda_m lambda_n dP and
    s(P) = sum_mn lambda_m(P) lambda_n(P) S_mn = int w'^2 dx:

        J = 1/2 sum_mn K_mn A_mn - 1/2 sum_mn S_mn B_mn + 1/8 int s(P)^2 dP,

    the membrane term integrated from s at the quadrature points in P (same
    value as 1/8 sum_mnpq S_mn S_pq int lambda_m lambda_n lambda_p lambda_q dP).

    Args:
        field_layout (FieldLayout): layout holding the interpolated monoms.
        cppgd (CPPGD): the decomposition, to look up the monom names.

    Returns:
        torch.Tensor: the scalar energy J.
    """
    directory = cppgd.directory()
    x_monoms = [field_layout[name] for name in directory["x"]]
    P_monoms = [field_layout[name] for name in directory["P"]]
    n_modes = len(x_monoms)

    # x part
    J_x = x_monoms[0].measure.values
    dw = [jacobian(m.x.values, m.u.values) for m in x_monoms]
    d2w = [hessian(m.x.values, m.u.values) for m in x_monoms]

    # P part
    J_P = P_monoms[0].measure.values
    P = P_monoms[0].x.values
    lmbdas = [m.u.values for m in P_monoms]

    bending, axial, s = 0.0, 0.0, 0.0
    for m in range(n_modes):
        for n in range(n_modes):
            K_mn = integrate(inner(d2w[m], d2w[n]) * J_x)
            S_mn = integrate(inner(dw[m], dw[n]) * J_x)
            A_mn = integrate(lmbdas[m] * lmbdas[n] * J_P)
            B_mn = integrate(P * lmbdas[m] * lmbdas[n] * J_P)
            bending = bending + K_mn * A_mn
            axial = axial + S_mn * B_mn
            s = s + lmbdas[m] * lmbdas[n] * S_mn  # at the quadrature points in P
    membrane = integrate(s**2 * J_P)

    return 0.5 * bending - 0.5 * axial + membrane / 8


# Post-processing.
#########################################################


def sample_monoms(cppgd, x, P):
    """Evaluate every monom of every mode on its own axis.

    Args:
        cppgd (CPPGD): the trained decomposition.
        x, P (torch.Tensor): sampling points, shape (N_x,) and (N_P,).

    Returns:
        tuple: x monoms (n_modes, N_x) and P monoms (n_modes, N_P), detached.
    """
    sampled = []
    for k, pts in enumerate((x, P)):
        spec = cppgd.monom_specs[k]
        sampled.append(torch.stack([
            PointWiseInterpolator(
                spec.space.mesh, spec.sf, cppgd.monoms[m][k], spec.space.mapping
            ).at_position(pts.reshape(-1, 1)).reshape(-1).detach()
            for m in range(cppgd.n_modes)
        ]))
    return tuple(sampled)


def main(
    *, n_elements, n_P, P_range, w_init, n_modes, adam_lr, adam_max_iter, tol,
    quadrature=None,
):
    """Build the separated problem, train it greedily and sample the result.

    Args:
        n_elements, n_P, P_range, w_init, quadrature: see `build_problem_pgd`.
        n_modes (int): number of greedy stages, one mode each.
        adam_lr (float): Adam learning rate.
        adam_max_iter (int): cap on the Adam iterations of each mode.
        tol (float): relative change of the loss that ends a mode.

    Returns:
        dict: training history, fine grids x and P, w(x, P) on the grid and
        the sampled monoms.
    """
    problem = build_problem_pgd(n_elements, n_P, P_range, w_init, quadrature=quadrature)
    model, cppgd = problem["model"], problem["cppgd"]

    print("* Training")
    trainer = GreedyTrainer(
        model,
        phases=[Phase(
            lambda params: torch.optim.Adam(params, lr=adam_lr),
            RelativeChange(tol=tol, max_iter=adam_max_iter),
            "adam",
        )],
        enrichment_criterion=MaxStages(n_modes),
    )
    history = trainer.enrich()
    for record in history.stages:
        print(
            f"mode {record.stage}: {record.n_iter} iterations, "
            f"{record.stop_reason}, energy = {record.energy:.6e}"
        )

    model.eval()
    x_fine = torch.linspace(0.0, 1.0, 201)
    P_fine = torch.linspace(P_range[0], P_range[1], 201)
    w_grid = cppgd.assemble([x_fine, P_fine]).detach()  # (N_x, N_P)
    x_monoms, P_monoms = sample_monoms(cppgd, x_fine, P_fine)

    return {
        "history": history,
        "x": x_fine,
        "P": P_fine,
        "w": w_grid,
        "x_monoms": x_monoms,
        "P_monoms": P_monoms,
    }


if __name__ == "__main__":
    from plotting import plot_loss_history, plot_monoms, plot_amplitude

    P1 = 4 * torch.pi**2  # first critical load, clamped-clamped

    # Numerical parameters
    N_ELEMENTS = 10
    N_P = 20
    P_RANGE = (0.5 * P1, 3 * P1)
    N_MODES = 3
    # Initial x monom (slopes by autograd), the P monom starts at 1
    W_INIT = symmetric_mode(eps=1e-2, m=1)

    # Minimization hyperparameters
    ADAM_LR = 1e-2
    ADAM_MAX_ITER = 5000  # cap per mode; a mode stops earlier on TOL
    TOL = 1e-6

    results = main(
        n_elements=N_ELEMENTS, n_P=N_P, P_range=P_RANGE, w_init=W_INIT,
        n_modes=N_MODES, adam_lr=ADAM_LR, adam_max_iter=ADAM_MAX_ITER, tol=TOL,
    )

    history = results["history"]
    plot_loss_history(
        history.losses,
        [record.stage for record in history.stages for _ in record.losses],
    ).show()
    plot_monoms(results["x"], results["x_monoms"], results["P"], results["P_monoms"]).show()
    mid = results["x"].shape[0] // 2  # x = 1/2
    plot_amplitude(results["P"], results["w"][mid], P_critical=P1).show()
