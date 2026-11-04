import numpy as np
import torch


# Import library modules
from neurom.quadratures import FourPoints1D
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
from neurom.training import GreedyTrainer, Phase, Stagnation, MaxStages
from neurom.math import jacobian, hessian, inner, integrate

from main import build_problem
from utils import rescale, grad_hessian


torch.set_default_dtype(torch.float64)


# Problem.
#########################################################


def x_noise(n_elements, amplitude, seed):
    """Gaussian noise on the Hermite DOFs (w, w') of an x monom.

    The 4 clamped DOFs are left at 0 (the Dirichlet constraint imposes them
    anyway).

    Args:
        n_elements (int): number of Hermite elements in x.
        amplitude (float): standard deviation of the noise.
        seed (int): seed of a dedicated `torch.Generator`.

    Returns:
        torch.Tensor: DOFs, shape (2 N, 1), intertwined (w, w').
    """
    N = n_elements + 1
    generator = torch.Generator().manual_seed(seed)
    values = amplitude * torch.randn(2 * N, 1, generator=generator)
    values[[0, 1, 2 * N - 2, 2 * N - 1]] = 0.0
    return values


def build_problem_pgd(n_elements, n_mu, mu_range, P, x_init, *, quadrature):
    """Assemble the separated clamped-clamped von Karman beam on a foundation w(x, mu).

    w(x, mu) = sum_m w_m(x) lambda_m(mu), x in [0, 1], mu = k^(1/4) in
    `mu_range`, k the stiffness of the Winkler foundation, P fixed.

    Args:
        n_elements (int): number of Hermite elements in x.
        n_mu (int): number of linear elements in mu.
        mu_range (tuple): (mu_min, mu_max).
        P (float): non-dimensional axial load.
        x_init (torch.Tensor): initial DOFs (w, w') of the x monom, shape
            (2 N, 1) (see `x_noise`). The mu monom starts at 1. Every new mode
            starts from the same values.
        quadrature (QuadratureRule): used on both axes; FourPoints1D() is exact
            for every integrand of the energy (degree 6 in x for M, degree 6
            in mu for C).

    Returns:
        dict: model, decomposition `cppgd`, field_layout, the node positions in
        x, the mu range and P.
    """
    quad = quadrature

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
        init_values=x_init,
        connectivity=connectivity_field,
    )

    ## Parameter mu
    N_mu = n_mu + 1
    mu_array = torch.linspace(mu_range[0], mu_range[1], N_mu).unsqueeze(-1)

    nodes_mu = torch.arange(0, N_mu)
    elements_mu = torch.vstack([torch.arange(0, N_mu - 1), torch.arange(1, N_mu)]).T

    connectivity_mu = Connectivity(nodes_mu, elements_mu)
    mu_positions = Field(name="mu_positions", connectivity=connectivity_mu, values=mu_array)
    mesh_mu = Mesh(connectivity=connectivity_mu, nodes_positions=mu_positions)

    space_mu = FactorSpace(
        name="mu",
        mesh=mesh_mu,
        mapping=IsoparametricMapping1D(LinearBar(), mesh_mu),
        quad=quad,
    )
    spec_mu = MonomSpec(
        space=space_mu,
        sf=LinearBar(),
        constraint=NoConstraint(),
        init_values=torch.ones(N_mu, 1),
    )

    ## Decomposition and model
    cppgd = CPPGD(monom_specs=[spec_x, spec_mu], name="pgd", n_modes_ini=1)

    # The domain holds no static assembly: the decomposition's own assemblies
    # are passed by the model at every forward.
    field_layout = FieldLayout()
    model = NeuROMModel(
        field_layout=field_layout,
        decomposition=cppgd,
        integration_domain=IntegrationDomain([]),
        loss=lambda out: energy(out, cppgd, P),
    )

    return {
        "model": model,
        "cppgd": cppgd,
        "field_layout": field_layout,
        "x_nodes": x_array.squeeze(-1),
        "mu_range": mu_range,
        "P": P,
    }


# Energy.
#########################################################


def energy(field_layout, cppgd, P):
    """Separated von Karman energy on the foundation, integrated over mu.

    With K_mn = int w_m'' w_n'' dx, S_mn = int w_m' w_n' dx, M_mn = int w_m w_n dx,
    A_mn = int lambda_m lambda_n dmu, C_mn = int mu^4 lambda_m lambda_n dmu and
    s(mu) = sum_mn lambda_m(mu) lambda_n(mu) S_mn = int w'^2 dx:

        J = 1/2 sum_mn K_mn A_mn - P/2 sum_mn S_mn A_mn + 1/8 int s(mu)^2 dmu
            + 1/2 sum_mn M_mn C_mn,

    i.e. J = int Pi(w(., mu)) dmu with k = mu^4, the membrane term integrated
    from s at the quadrature points in mu.

    Args:
        field_layout (FieldLayout): layout holding the interpolated monoms.
        cppgd (CPPGD): the decomposition, to look up the monom names.
        P (float): non-dimensional axial load.

    Returns:
        torch.Tensor: the scalar energy J.
    """
    directory = cppgd.directory()
    x_monoms = [field_layout[name] for name in directory["x"]]
    mu_monoms = [field_layout[name] for name in directory["mu"]]
    n_modes = len(x_monoms)

    # x part
    J_x = x_monoms[0].measure.values
    w = [m.u.values for m in x_monoms]
    dw = [jacobian(m.x.values, m.u.values) for m in x_monoms]
    d2w = [hessian(m.x.values, m.u.values) for m in x_monoms]

    # mu part
    J_mu = mu_monoms[0].measure.values
    mu = mu_monoms[0].x.values
    lmbdas = [m.u.values for m in mu_monoms]

    bending, axial, foundation, s = 0.0, 0.0, 0.0, 0.0
    for m in range(n_modes):
        for n in range(n_modes):
            K_mn = integrate(inner(d2w[m], d2w[n]) * J_x)
            S_mn = integrate(inner(dw[m], dw[n]) * J_x)
            M_mn = integrate(w[m] * w[n] * J_x)
            A_mn = integrate(lmbdas[m] * lmbdas[n] * J_mu)
            C_mn = integrate(mu**4 * lmbdas[m] * lmbdas[n] * J_mu)
            bending = bending + K_mn * A_mn
            axial = axial + S_mn * A_mn
            foundation = foundation + M_mn * C_mn
            s = s + lmbdas[m] * lmbdas[n] * S_mn  # at the quadrature points in mu
    membrane = integrate(s**2 * J_mu)

    return 0.5 * bending - 0.5 * P * axial + membrane / 8 + 0.5 * foundation


# Post-processing.
#########################################################


def sample_monoms(cppgd, x, mu):
    """Evaluate every monom of every mode on its own axis.

    Args:
        cppgd (CPPGD): the trained decomposition.
        x, mu (torch.Tensor): sampling points, shape (N_x,) and (N_mu,).

    Returns:
        tuple: x monoms (n_modes, N_x) and mu monoms (n_modes, N_mu), detached.
    """
    sampled = []
    for k, pts in enumerate((x, mu)):
        spec = cppgd.monom_specs[k]
        sampled.append(torch.stack([
            PointWiseInterpolator(
                spec.space.mesh, spec.sf, cppgd.monoms[m][k], spec.space.mapping
            ).at_position(pts.reshape(-1, 1)).reshape(-1).detach()
            for m in range(cppgd.n_modes)
        ]))
    return tuple(sampled)


def axial_displacement(field_layout, cppgd, x_nodes, mu_monoms, P):
    """Non-dimensional axial displacement u(x, mu) at the mesh nodes in x.

    u(x_j, mu) = -[(P - S(mu)/2) x_j + 1/2 int_0^{x_j} w'^2], with
    int_0^{x} w'^2 = sum_mn lambda_m(mu) lambda_n(mu) C_mn(x),
    C_mn(x) = int_0^x w_m' w_n' and S(mu) the same at x = 1, so that
    u_hat = (r^2 / l) u. C_mn reuses the quadrature in x of the energy.

    Args:
        field_layout (FieldLayout): layout holding the up-to-date quadrature
            results of the x monoms.
        cppgd (CPPGD): the decomposition, to look up the monom names.
        x_nodes (torch.Tensor): node positions in x, shape (N_nodes,).
        mu_monoms (torch.Tensor): mu monoms sampled at the wanted mu, shape
            (n_modes, N_mu).
        P (float): non-dimensional axial load.

    Returns:
        torch.Tensor: non-dimensional axial displacement, shape (N_nodes, N_mu).
    """
    x_monoms = [field_layout[name] for name in cppgd.directory()["x"]]
    J_x = x_monoms[0].measure.values
    dw = [jacobian(m.x.values, m.u.values) for m in x_monoms]
    n_elements = x_nodes.shape[0] - 1

    # int_0^{x_j} w'^2 at every node and every mu
    cum = torch.zeros(x_nodes.shape[0], mu_monoms.shape[1])
    for m in range(len(x_monoms)):
        for n in range(len(x_monoms)):
            # Per-element integrals of w_m' w_n', then cumulative from x = 0
            I_e = (inner(dw[m], dw[n]) * J_x).detach().reshape(n_elements, -1).sum(dim=1)
            C_mn = torch.cat([torch.zeros(1), torch.cumsum(I_e, dim=0)])
            cum = cum + C_mn[:, None] * (mu_monoms[m] * mu_monoms[n])[None, :]
    S = cum[-1]

    return -((P - 0.5 * S)[None, :] * x_nodes[:, None] + 0.5 * cum)


def pgd_dofs(cppgd, mu):
    """Free DOFs of w_PGD(., mu) = sum_m w_m lambda_m(mu), all the modes.

    Args:
        cppgd (CPPGD): the trained decomposition.
        mu (torch.Tensor): parameter values, shape (N_mu,).

    Returns:
        torch.Tensor: shape (n_free, N_mu), ordered as `values_reduced` of the
        non-decomposed problem (`main.build_problem`, same Dirichlet nodes).
    """
    _, mu_monoms = sample_monoms(cppgd, torch.zeros(1), mu)  # (n_modes, N_mu)
    q_m = torch.stack([row[0].values_reduced.detach().reshape(-1) for row in cppgd.monoms])
    return q_m.T @ mu_monoms


def lambda_min(cppgd, mu, P, n_elements, *, quadrature):
    """Smallest eigenvalue of the Hessian of Pi at w_PGD(., mu), k = mu^4.

    Pi is the energy of the non-decomposed problem (`main.build_problem`) on
    the mesh of the PGD: lambda_min > 0 on a minimum, <= 0 on a saddle.

    Args:
        cppgd (CPPGD): the trained decomposition.
        mu (torch.Tensor): parameter values, shape (N_mu,).
        P (float): non-dimensional axial load.
        n_elements (int): number of Hermite elements of the PGD in x.
        quadrature (QuadratureRule): the quadrature of the PGD.

    Returns:
        torch.Tensor: lambda_min, shape (N_mu,).
    """
    Q = pgd_dofs(cppgd, mu)
    values = []
    for i, m in enumerate(mu.tolist()):
        model = build_problem(n_elements, P, k=m**4, quadrature=quadrature)["model"]
        (p,) = model.parameters()
        with torch.no_grad():
            p.copy_(Q[:, i].view_as(p))
        _, _, H = grad_hessian(model)
        values.append(torch.linalg.eigvalsh(0.5 * (H + H.T))[0])
    return torch.stack(values)


# Comparison with a reference.
#########################################################


def reference_values(Q, n_elements, P, x, *, per_solution):
    """Deflections of non-decomposed solutions, given by their free DOFs.

    Args:
        Q (torch.Tensor): free DOFs of the solutions, shape (n_free, N_i), on
            a mesh of `n_elements` Hermite elements (`main.build_problem`).
        n_elements (int): number of elements of that mesh.
        P (float): non-dimensional axial load (does not change w).
        x (torch.Tensor): points in x, shape (N_i,) if `per_solution` (x_i for
            solution i), else (N_x,) (the same points for all).

    Returns:
        torch.Tensor: shape (N_i,) if `per_solution`, else (N_x, N_i).
    """
    problem = build_problem(n_elements, P)
    (p,) = problem["model"].parameters()
    pwi = PointWiseInterpolator(problem["mesh"], problem["sf"], problem["u"], problem["mapping"])
    values = []
    for i in range(Q.shape[1]):
        with torch.no_grad():
            p.copy_(Q[:, i].view_as(p))
        pts = x[i : i + 1] if per_solution else x
        values.append(pwi.at_position(pts.unsqueeze(-1)).reshape(-1).detach())
    values = torch.stack(values, dim=1)
    return values[0] if per_solution else values


def align_signs(cppgd, Q, mu, n_elements_ref, P, *, M=None, n_grid=2001):
    """s_i = sign of int_0^1 w_PGD(x, mu_i) w_ref(x, mu_i) dx, all the modes.

    The branches are defined up to their sign: s_i puts the reference on the
    side taken by the PGD. The integral uses the full fields in x: q_PGD^T M q_ref
    when the mass matrix M is given and the meshes are the same (exact), the
    trapezoidal rule on `n_grid` points otherwise. s_i = 1 when it is 0.

    Args:
        cppgd (CPPGD): the trained decomposition.
        Q (torch.Tensor): free DOFs of the reference, shape (n_free, N_i).
        mu (torch.Tensor): parameter of each reference solution, shape (N_i,).
        n_elements_ref (int): number of elements of the reference mesh.
        P (float): non-dimensional axial load.
        M (torch.Tensor, optional): Hermite mass matrix on the free DOFs of
            the reference mesh, int w v dx = q^T M q_v.
        n_grid (int): number of points of the grid in x when M is not used.

    Returns:
        torch.Tensor: s, shape (N_i,), entries +-1.
    """
    Q_pgd = pgd_dofs(cppgd, mu)
    if M is not None and Q_pgd.shape[0] == Q.shape[0]:
        inner_x = ((M @ Q_pgd) * Q).sum(dim=0)
    else:
        x = torch.linspace(0.0, 1.0, n_grid)
        x_monoms, mu_monoms = sample_monoms(cppgd, x, mu)
        W_pgd = x_monoms.T @ mu_monoms  # (n_grid, N_i)
        W_ref = reference_values(Q, n_elements_ref, P, x, per_solution=False)
        inner_x = torch.trapezoid(W_pgd * W_ref, x, dim=0)
    s = torch.sign(inner_x)
    s[s == 0] = 1.0
    return s


def errors(cppgd, x, mu, w_ref, s):
    """Relative L2 error e(n) of the PGD truncated to n modes, on sample points.

    e(n)^2 = sum_i (w_n(x_i, mu_i) - s_i w_ref_i)^2 / sum_i w_ref_i^2, the
    Monte-Carlo estimate of the relative L2 error on the box when the points
    (x_i, mu_i) are drawn uniformly (e.g. by LHS); w_n uses the first n modes.

    Args:
        cppgd (CPPGD): the trained decomposition.
        x, mu (torch.Tensor): sample points, shape (N_i,) each.
        w_ref (torch.Tensor): reference at the sample points, shape (N_i,).
        s (torch.Tensor): signs of `align_signs`, shape (N_i,).

    Returns:
        torch.Tensor: e, shape (n_modes + 1,), e[0] = 1.
    """
    x_monoms, mu_monoms = sample_monoms(cppgd, x, mu)  # (n_modes, N_i)
    w_n = torch.cumsum(x_monoms * mu_monoms, dim=0)
    w_n = torch.cat([torch.zeros(1, w_n.shape[1]), w_n])
    target = s * w_ref
    return (((w_n - target) ** 2).sum(dim=1) / (target**2).sum()).sqrt()


def save_decomposition(cppgd, path, **parameters):
    """Write the DOFs of every monom and the parameters of the run.

    Args:
        cppgd (CPPGD): the decomposition.
        path (Path): output file (.pt).
        **parameters: stored as they are, e.g. the arguments of
            `build_problem_pgd` needed by `load_decomposition`.
    """
    torch.save({
        "x_monoms": [row[0].full_values().detach() for row in cppgd.monoms],
        "mu_monoms": [row[1].full_values().detach() for row in cppgd.monoms],
        **parameters,
    }, path)


def load_decomposition(path, *, quadrature):
    """Rebuild a decomposition written by `save_decomposition`.

    Args:
        path (Path): file of `save_decomposition`, holding n_elements, n_mu,
            mu_range and P.
        quadrature (QuadratureRule): the quadrature of the run.

    Returns:
        tuple: the CPPGD, the stored dict and the problem, whose field_layout
        holds the quadrature results of the loaded DOFs (for `axial_displacement`).
    """
    saved = torch.load(path)
    problem = build_problem_pgd(
        saved["n_elements"], saved["n_mu"], saved["mu_range"], saved["P"],
        torch.zeros(2 * (saved["n_elements"] + 1), 1), quadrature=quadrature,
    )
    cppgd = problem["cppgd"]
    for _ in range(len(saved["x_monoms"]) - 1):
        cppgd.add_mode()
    cppgd.register_into(problem["field_layout"])
    with torch.no_grad():
        for row, q_x, q_mu in zip(cppgd.monoms, saved["x_monoms"], saved["mu_monoms"]):
            row[0].values_reduced.copy_(q_x[row[0].dofs_free])
            row[1].values_reduced.copy_(q_mu[row[1].dofs_free])

    # Fill the quadrature results with the loaded DOFs
    problem["model"].train()
    problem["model"]()
    return cppgd, saved, problem


def default_phases(*, adam_lr, adam_max_iter, lbfgs_lr, lbfgs_max_iter, tol):
    """Adam, then optionally LBFGS (strong Wolfe), each ended at the first stagnating
    iteration (`Stagnation` without patience).

    Args:
        adam_lr (float): Adam learning rate.
        adam_max_iter (int): cap on the Adam iterations of each mode.
        lbfgs_lr (float): LBFGS learning rate.
        lbfgs_max_iter (int): cap on the LBFGS iterations run after Adam on
            each mode; 0 disables the LBFGS phase.
        tol (float): relative change of the loss that ends a phase.

    Returns:
        list[Phase]: the phases run on every mode.
    """
    phases = [Phase(
        lambda params: torch.optim.Adam(params, lr=adam_lr),
        Stagnation(tol=tol, patience=0, no_improvement=None, max_iter=adam_max_iter),
        "adam",
    )]
    if lbfgs_max_iter > 0:
        phases.append(Phase(
            lambda params: torch.optim.LBFGS(
                params, lr=lbfgs_lr, line_search_fn="strong_wolfe"
            ),
            Stagnation(
                tol=tol, patience=0, no_improvement=None, max_iter=lbfgs_max_iter
            ),
            "lbfgs",
        ))
    return phases


def main(
    *, n_elements, n_mu, mu_range, P, noise_amplitude, seed, phases,
    enrichment_criterion, E, A, I, length, quadrature, trainer=GreedyTrainer,
    progress=None,
):
    """Build the separated problem, train it and sample the result.

    Args:
        n_elements, n_mu, mu_range, P, quadrature: see `build_problem_pgd`.
        noise_amplitude, seed (float, int): initial x monom, see `x_noise`.
        phases (list[Phase]): optimizer phases run on every mode, each with
            its own stopping criterion (see `default_phases`).
        enrichment_criterion (EnrichmentCriterion): decides when to stop
            adding modes, e.g. `MaxStages(n_modes)`.
        E, A, I, length (float): physical constants, used by `rescale`.
        trainer (type): training strategy, `GreedyTrainer` (earlier modes
            frozen) or `SimultaneousTrainer` (all the modes retrained).
        progress (ProgressReporter, optional): reports the training as it
            goes, e.g. `ProgressLog(every=500)`; None prints a summary at the end.

    Returns:
        dict: training history, the decomposition, the sampled monoms, and on
        the grids (x nodes or fine x) x (fine mu) the deflection w and axial
        displacement, non-dimensional and dimensional (suffix `_hat`).
    """
    x_init = x_noise(n_elements, noise_amplitude, seed)
    problem = build_problem_pgd(
        n_elements, n_mu, mu_range, P, x_init, quadrature=quadrature
    )
    model, cppgd = problem["model"], problem["cppgd"]

    print("* Training")
    history = trainer(
        model, phases=phases, enrichment_criterion=enrichment_criterion,
        progress=progress,
    ).enrich()
    if progress is None:
        for record in history.stages:
            print(
                f"mode {record.stage}: {record.n_iter} iterations, "
                f"{record.stop_reason}, energy = {record.energy:.6e}"
            )

    # Refresh the quadrature results with the trained DOFs
    model.train()
    model()

    x_nodes = problem["x_nodes"]
    x_fine = torch.linspace(0.0, 1.0, 201)
    mu_fine = torch.linspace(mu_range[0], mu_range[1], 201)
    x_monoms, mu_monoms = sample_monoms(cppgd, x_fine, mu_fine)
    u_axial = axial_displacement(problem["field_layout"], cppgd, x_nodes, mu_monoms, P)

    model.eval()
    w_grid = cppgd.assemble([x_fine, mu_fine]).detach()  # (N_x, N_mu)
    w_nodes = cppgd.assemble([x_nodes, mu_fine]).detach()

    # Axial displacement linearly interpolated from the nodes, for every mu
    u_fine = torch.from_numpy(np.stack(
        [np.interp(x_fine.numpy(), x_nodes.numpy(), u_mu) for u_mu in u_axial.T.numpy()],
        axis=1,
    )).to(x_fine.dtype)

    x_hat, w_hat, u_hat, _ = rescale(
        x_nodes, w_nodes, u_axial, P, E=E, A=A, I=I, length=length
    )
    x_fine_hat, w_fine_hat, u_fine_hat, _ = rescale(
        x_fine, w_grid, u_fine, P, E=E, A=A, I=I, length=length
    )

    return {
        "history": history,
        "cppgd": cppgd,
        "problem": problem,
        "x": x_fine,
        "mu": mu_fine,
        "w": w_grid,
        "u_fine": u_fine,
        "x_nodes": x_nodes,
        "w_nodes": w_nodes,
        "u_axial": u_axial,
        "x_monoms": x_monoms,
        "mu_monoms": mu_monoms,
        "x_hat": x_hat,
        "w_hat": w_hat,
        "u_hat": u_hat,
        "x_fine_hat": x_fine_hat,
        "w_fine_hat": w_fine_hat,
        "u_fine_hat": u_fine_hat,
    }


if __name__ == "__main__":
    from plotting import plot_loss_history, plot_monoms

    # Physical constants (steel, 1 cm x 1 cm square section)
    E = 210e9  # Young's modulus [Pa]
    b = 1e-2  # section side [m]
    A = b**2  # cross-section area [m^2]
    I = b**4 / 12  # second moment of area [m^4]
    L = 1.0  # beam length [m]

    # Numerical parameters (Exp-pgd-mu-1, « Choix initial »)
    P = 2200.0
    N_ELEMENTS = 20
    N_MU = 20
    MU_RANGE = (0.0, 40.0)
    N_MODES = 15
    # Initial x monom: Gaussian noise on the free DOFs, the mu monom starts at 1
    NOISE = 1e-2
    SEED = 0

    # Minimization hyperparameters
    ADAM_LR = 1e-2
    ADAM_MAX_ITER = 5000  # cap per mode; a mode stops earlier on TOL
    LBFGS_LR = 1.0
    LBFGS_MAX_ITER = 10  # LBFGS after Adam on each mode, 0 disables it
    TOL = 1e-6

    results = main(
        n_elements=N_ELEMENTS, n_mu=N_MU, mu_range=MU_RANGE, P=P,
        noise_amplitude=NOISE, seed=SEED,
        phases=default_phases(
            adam_lr=ADAM_LR, adam_max_iter=ADAM_MAX_ITER,
            lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER, tol=TOL,
        ),
        enrichment_criterion=MaxStages(N_MODES),
        E=E, A=A, I=I, length=L, quadrature=FourPoints1D(),
    )

    history = results["history"]
    plot_loss_history(
        history.losses,
        [record.stage for record in history.stages for _ in record.losses],
    ).show()
    plot_monoms(results["x"], results["x_monoms"], results["mu"], results["mu_monoms"]).show()
