"""Five-parametric 1D bar with a tanh-graded Young's modulus -- CP-PGD setup.

Bi-clamped bar under a constant axial load, whose modulus has two zones E1 and E2
separated by a tanh transition centred at ``alpha`` with slope ``n``:

    E(x, E1, E2, alpha, n) = (E2 - E1) / 2 * tanh(n (x - alpha)) + (E2 + E1) / 2

The four material/geometry parameters are treated as extra coordinates, so the
solution is sought in separated form

    u(x, E1, E2, alpha, n) = sum_i X_i(x) lambda_i(E1) mu_i(E2) A_i(alpha) N_i(n)

Because ``E1, E2 > 0`` and ``tanh`` maps into (-1, 1), the modulus stays strictly
between E1 and E2 -- positivity is structural, no clamping needed.

This module builds the problem (axes, decomposition, energy, model), evaluates
the energy once, and then trains it greedily in ``main`` -- one mode per stage,
every earlier mode frozen -- reporting a per-stage diagnostics table. Pass
``train=False`` for assembly only, ``plot=False`` to skip the figures.

Accuracy is judged against ``reference_fem_solution.py``: a direct, non-reduced
FEM solve at a handful of parameter points, saved to ``reference_solution.pt``.
Generate it once before plotting::

    python docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py
"""

from dataclasses import dataclass

import torch

from neurom.constraints import Dirichlet, NoConstraint
from neurom.decompositions import Axis, CPPGD
from neurom.differential import jacobian_field
from neurom.field_layout import FieldLayout
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.inner import inner
from neurom.integrate import integrate
from neurom.interpolation.integration_domain import IntegrationDomain
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.meshes import Topology
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import MidPoint1D
from neurom.shape_functions import LinearSegment
from neurom.training import GreedyTrainer, RelativeChange, RelativeGain

torch.set_default_dtype(torch.float32)

# --- domain and parameter intervals ---------------------------------------
X_MIN, X_MAX = 0.0, 10.0
E1_MIN, E1_MAX = 10.0, 100.0
E2_MIN, E2_MAX = 10.0, 100.0
# alpha is kept away from the clamped ends so the transition always sits inside
# the bar; n spans near-uniform (tanh almost linear over the bar) to a sharp
# interface roughly one length unit wide.
ALPHA_MIN, ALPHA_MAX = 2.0, 8.0
N_MIN, N_MAX = 0.5, 5.0

DEFAULT_N_NODES = {"space": 30, "E1": 20, "E2": 20, "alpha": 15, "n": 15}

LOAD_VALUE = 1000.0
# The 2-parameter case was exactly rank-1; a non-separable modulus has no reason
# to be low-rank, hence the larger budget.
N_MODES_MAX = 10

# Axis order is load-bearing: it fixes the column order of CPPGD.evaluate and
# the key order of CPPGD.directory().
AXIS_ORDER = ["space", "E1", "E2", "alpha", "n"]

AXIS_BOUNDS = {
    "space": (X_MIN, X_MAX),
    "E1": (E1_MIN, E1_MAX),
    "E2": (E2_MIN, E2_MAX),
    "alpha": (ALPHA_MIN, ALPHA_MAX),
    "n": (N_MIN, N_MAX),
}


def make_axis(name, lo, hi, n_nodes, constraint, sf, quad, mapping, init_value=0.5):
    """Build one Axis on a uniform 1-D mesh of ``n_nodes`` nodes over [lo, hi].

    Wraps the Topology / Field / Axis boilerplate so the five axes are not five
    copy-pasted blocks. The Axis builds its own Mesh and QuadratureContext.

    Args:
        name (str): Axis name; also the prefix of its nodes-positions field.
        lo, hi (float): Interval bounds.
        n_nodes (int): Number of mesh nodes (so ``n_nodes - 1`` linear elements).
        constraint (Constraint): Dirichlet or NoConstraint for the monoms.
        sf (ShapeFunction), quad (QuadratureRule), mapping: shared discretisation.
        init_value (float): Constant seed for every new monom's nodal values.

    Returns:
        Axis: ready to be handed to CPPGD.
    """
    positions = torch.linspace(lo, hi, n_nodes).unsqueeze(-1)
    nodes = torch.arange(0, n_nodes)
    elements = torch.vstack([torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)]).T
    topology = Topology(nodes, elements)
    nodes_positions = Field(
        name=f"{name}_positions", topology=topology, values=positions
    )
    return Axis(
        name=name,
        nodes_positions=nodes_positions,
        sf=sf,
        mapping=mapping,
        quad=quad,
        constraint=constraint,
        init_values=init_value * torch.ones(n_nodes, 1),
    )


@dataclass
class Problem:
    """Everything ``build_problem`` assembles, kept together for the caller.

    Attributes:
        model (NeuROMModel): the trainable model (``model()`` fills the layout).
        pgd (CPPGD): the separated representation.
        field_layout (FieldLayout): holds the monom fields and the load.
        domain (IntegrationDomain): interpolates every active field.
        axes (dict[str, Axis]): the five axes, keyed by name.
        history (TrainingHistory | None): Filled by ``main`` when it trains;
            None when the problem is only assembled.
    """

    model: NeuROMModel
    pgd: CPPGD
    field_layout: FieldLayout
    domain: IntegrationDomain
    axes: dict[str, Axis]
    history: object = None


def build_problem(
    loss_fn, *, n_modes_max=N_MODES_MAX, n_modes_ini=1, n_nodes=None, quad=None
):
    """Assemble the five axes, the CP-PGD, the load and the model.

    The energy is *injected*: this function never references ``energy`` directly,
    so the same wiring drives a different (e.g. non-linear PGD) functional
    unchanged.

    Args:
        loss_fn (Callable): ``loss_fn(field_layout, decomposition) -> Tensor``.
        n_modes_max (int): Mode budget of the CP-PGD.
        n_modes_ini (int): Number of initially active (trainable) modes.
        n_nodes (dict[str, int], optional): Per-axis node counts overriding
            ``DEFAULT_N_NODES``; used by the tests to build a tiny problem.
        quad (QuadratureRule, optional): Shared quadrature rule for every axis;
            defaults to ``MidPoint1D()`` (one point per element). Injectable so
            tests can exercise ``N_q > 1`` rules (e.g. ``TwoPoints1D``), which
            catch broadcasting bugs that a single quadrature point hides.

    Returns:
        Problem: the assembled objects.
    """
    sf = LinearSegment()
    quad = quad if quad is not None else MidPoint1D()
    mapping = IsoparametricMapping1D(sf)

    counts = dict(DEFAULT_N_NODES)
    if n_nodes is not None:
        counts.update(n_nodes)

    field_layout = FieldLayout()

    n_x = counts["space"]
    constraints = {
        "space": Dirichlet(nodes=[0, n_x - 1], values_imposed=torch.zeros(2, 1)),
        "E1": NoConstraint(),
        "E2": NoConstraint(),
        "alpha": NoConstraint(),
        "n": NoConstraint(),
    }
    axes = [
        make_axis(
            name,
            *AXIS_BOUNDS[name],
            counts[name],
            constraints[name],
            sf,
            quad,
            mapping,
        )
        for name in AXIS_ORDER
    ]

    pgd = CPPGD(axes=axes, n_modes_max=n_modes_max, name="pgd", n_modes_ini=n_modes_ini)

    # The load is a *field* f(x), interpolated on the SAME quadrature as the
    # space monoms so that inner(f, X) aligns point-for-point. It is the single
    # spatial factor of the separated source f = f_0(x) (x) 1(E1) (x) ... ; the
    # constant parametric factors are carried by the int lambda_i dE1 ... terms
    # of the load part of the energy.
    axis_space = next(axis for axis in axes if axis.name == "space")
    load_field = field_layout.add(
        Field(
            name="load",
            topology=axis_space.topology,
            values=LOAD_VALUE * torch.ones(n_x, 1),
        )
    )
    assembly_f = QuadratureAssembly(axis_space.context, sf, load_field)

    domain = IntegrationDomain([*pgd.assemblies(), assembly_f])

    model = NeuROMModel(
        field_layout=field_layout,
        decomposition=pgd,
        integration_domain=domain,
        loss=lambda layout: loss_fn(layout, pgd),
    )

    return Problem(
        model=model,
        pgd=pgd,
        field_layout=field_layout,
        domain=domain,
        axes={axis.name: axis for axis in axes},
    )


def energy(field_layout, decomposition, load_name="load"):
    """Total potential energy of the tanh-graded bar, in separated form.

    With u = sum_i X_i(x) lambda_i(E1) mu_i(E2) A_i(alpha) N_i(n) and

        E(x, E1, E2, alpha, n) = (E2 + E1)/2 + (E2 - E1)/2 tanh(n (x - alpha)),

    every factor of the elastic term separates into a product of 1-D integrals
    *except* tanh(n (x - alpha)), which couples x, alpha and n. That coupled block
    is integrated by an exact 3-D tensor-product quadrature over those three
    axes' quadrature points (one einsum per mode pair); the (E2 +/- E1)/2
    prefactors stay 1-D moments of the E1 and E2 factors.

    Sign convention follows the 2-parameter example: the returned value is
    ``elastic + load``, the load field carrying its own sign.

    Args:
        field_layout (FieldLayout): filled by ``model()`` (train mode).
        decomposition (CPPGD): supplies the active monom names via ``directory()``.
        load_name (str): name of the load field in the layout.

    Returns:
        torch.Tensor: 0-dim energy.
    """
    directory = decomposition.directory()
    n_modes = len(directory["space"])

    spc = [field_layout[name] for name in directory["space"]]
    e1 = [field_layout[name] for name in directory["E1"]]
    e2 = [field_layout[name] for name in directory["E2"]]
    alp = [field_layout[name] for name in directory["alpha"]]
    slp = [field_layout[name] for name in directory["n"]]

    # Space factors. jacobian_field returns (N_e, N_q, *u_shape, d) -- one extra
    # trailing axis compared to u -- which must be *contracted away* by inner(),
    # not reshaped away: reshaping only appears to work for d = 1 and turns the
    # cross terms into a silent element-wise broadcast (see the 2-parameter
    # example for the full story).
    X = [r.u for r in spc]
    gX = [jacobian_field(x=r.x, u=r.u) for r in spc]
    Jx = [r.measure for r in spc]

    # Parametric factors, with their own coordinate values (needed for the
    # first moments int E1 lambda_i lambda_j dE1 and int E2 mu_i mu_j dE2).
    lam, E1_val, J1 = [r.u for r in e1], [r.x for r in e1], [r.measure for r in e1]
    mu, E2_val, J2 = [r.u for r in e2], [r.x for r in e2], [r.measure for r in e2]
    A, Ja = [r.u for r in alp], [r.measure for r in alp]
    N, Jn = [r.u for r in slp], [r.measure for r in slp]

    # NB: as in the 2-parameter example, the cross terms (i, j) assume modes i
    # and j share a mesh per axis (the measure and coordinates indexed by i are
    # used for both). Independent per-mode meshes would need a common
    # intersection mesh with a recomputed measure. tanh_grid below is the more
    # fragile consumer of this assumption, since it hard-codes mode 0's
    # quadrature points for every (i, j) pair.
    #
    # The one non-separable block: tanh(n (x - alpha)) on the tensor product of
    # the space, alpha and n quadrature points, shape (Qx, Qalpha, Qn). It does
    # not depend on the mode pair, so it is built once and reused below.
    # Rebuilt every call (rather than cached at setup) so it stays correct if the
    # meshes ever become trainable (r-adaptivity).
    xq = spc[0].x.reshape(-1)
    aq = alp[0].x.reshape(-1)
    nq = slp[0].x.reshape(-1)
    tanh_grid = torch.tanh(nq[None, None, :] * (xq[:, None, None] - aq[None, :, None]))

    elastic = 0.0
    for i in range(n_modes):
        for j in range(n_modes):
            # Densities at the quadrature points, reused both integrated (the
            # separable part) and raw (contracted against tanh_grid).
            kx_density = inner(gX[i], gX[j]) * Jx[i]
            a_density = A[i] * A[j] * Ja[i]
            n_density = N[i] * N[j] * Jn[i]

            Kx = integrate(kx_density)
            P0 = integrate(a_density)
            Q0 = integrate(n_density)
            L0 = integrate(lam[i] * lam[j] * J1[i])
            L1 = integrate(E1_val[i] * lam[i] * lam[j] * J1[i])
            M0 = integrate(mu[i] * mu[j] * J2[i])
            M1 = integrate(E2_val[i] * mu[i] * mu[j] * J2[i])

            # kx_density, a_density and n_density are all scalar (N_e, N_q, 1)
            # densities (no trailing vector/jacobian axis like gX), so
            # .reshape(-1) is an exact flatten (numel == Qx, Qalpha, Qn
            # respectively) -- unlike the d-trailing jacobian_field output
            # discussed above, there is no broadcast trap here.
            T = torch.einsum(
                "xan,x,a,n->",
                tanh_grid,
                kx_density.reshape(-1),
                a_density.reshape(-1),
                n_density.reshape(-1),
            )

            mean_modulus = 0.5 * (M1 * L0 + L1 * M0)  # (E2 + E1) / 2
            half_contrast = 0.5 * (M1 * L0 - L1 * M0)  # (E2 - E1) / 2
            elastic = elastic + mean_modulus * Kx * P0 * Q0 + half_contrast * T
    elastic = 0.5 * elastic

    # Constant load: separable across every axis, so one 1-D integral per factor.
    load_f = field_layout[load_name].u
    load = 0.0
    for i in range(n_modes):
        Fx = integrate(inner(load_f, X[i]) * Jx[i])
        load = load + (
            Fx
            * integrate(lam[i] * J1[i])
            * integrate(mu[i] * J2[i])
            * integrate(A[i] * Ja[i])
            * integrate(N[i] * Jn[i])
        )

    return elastic + load


## Reference solution and plotting helpers


def modulus(x, E1, E2, alpha, n):
    """The tanh-graded modulus E(x, E1, E2, alpha, n), broadcasting over ``x``.

    Single source of truth for the modulus law: the separated ``energy`` above
    hard-codes its tanh structure to keep the integrals separable, and
    ``reference_fem_solution.py`` injects *this* function into a direct FEM
    solve. If the law changes, both must change together.
    """
    return 0.5 * (E2 - E1) * torch.tanh(n * (x - alpha)) + 0.5 * (E2 + E1)


def load_reference_bundle():
    """Load the FEM reference solutions from the sibling ``reference_fem_solution``.

    Imported by path, and lazily, so that this example neither depends on the
    reference module at import time nor requires the reference file to exist
    unless you actually plot.

    Returns:
        dict: the bundle described in ``reference_fem_solution.generate``.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "reference_fem_solution.py"
    spec = importlib.util.spec_from_file_location("reference_fem_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reference()


def plot_convergence(history, save_path="pgd5_convergence.png"):
    """Plot the energy against the training iteration, with the stage boundaries.

    Args:
        history (TrainingHistory): filled by ``GreedyTrainer.enrich``.
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history.losses)), history.losses, "b-")

    # One vertical rule per greedy stage boundary: each stage adds one mode, so
    # these mark where the energy is allowed to drop again.
    boundary = 0
    for record in history.stages[:-1]:
        boundary += record.n_iter
        ax.axvline(boundary, color="grey", ls=":", lw=1)

    ax.set_title("Greedy training convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy (loss)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()


def relative_errors(pgd, reference=None):
    """Relative L2 errors of the PGD against the stored FEM reference.

    Two different things are reported, because they answer different questions:

    * ``per_point[label]`` is a **space-only** error at one parameter point --
      ``||u_pgd(., p) - u_ref(., p)|| / ||u_ref(., p)||`` over the reference
      ``x`` grid, with the parameters frozen at ``p``. This is what a single
      panel of ``plot_solution`` shows.
    * ``overall`` is the error over space **and** the sampled parameter points
      jointly: the two ``(K, P)`` tables are flattened and one ratio of norms is
      taken. It is dominated by the points with the largest deflection (the soft
      ones), which is the honest global figure -- the mean of the per-point
      errors, by contrast, weights a barely-loaded stiff bar as much as a soft
      one. Both are returned; neither weights the parameter volume, since the
      reference points are only a handful of samples, not a quadrature.

    Args:
        pgd (CPPGD): the trained decomposition.
        reference (dict, optional): a loaded reference bundle; by default the
            one on disk next to this script.

    Returns:
        dict: ``{"per_point": {label: float}, "overall": float, "u_pgd": Tensor}``
        where ``u_pgd`` is the ``(K, P)`` table the errors were computed from.
    """
    if reference is None:
        reference = load_reference_bundle()

    x_ref = reference["x"]
    u_ref = reference["u"]

    rows = []
    for row in reference["params"]:
        params = dict(zip(reference["param_names"], (v.item() for v in row)))
        # (P, 5) query matrix: the reference x grid, the parameters held constant.
        columns = {"space": x_ref}
        columns.update({name: torch.full_like(x_ref, v) for name, v in params.items()})
        rows.append(
            pgd.evaluate(
                torch.stack([columns[name] for name in AXIS_ORDER], dim=1)
            ).reshape(-1)
        )
    u_pgd = torch.stack(rows)

    per_point = {
        label: (
            torch.linalg.norm(u_pgd[k] - u_ref[k]) / torch.linalg.norm(u_ref[k])
        ).item()
        for k, label in enumerate(reference["labels"])
    }
    overall = (torch.linalg.norm(u_pgd - u_ref) / torch.linalg.norm(u_ref)).item()
    return {"per_point": per_point, "overall": overall, "u_pgd": u_pgd}


def plot_solution(pgd, reference=None, labels=None, save_path="pgd5_vs_reference.png"):
    """Compare the PGD to the FEM reference at the two hardest parameter points.

    Ten superposed near-parabolas say very little; two well-chosen points say a
    lot. The default pair is the one the reference bundle flags as
    ``metadata["highlight"]``: sharp, high-contrast moduli of opposite sign and
    different ratio, where ``u`` has a visible kink and a separated
    representation is under real strain.

    Each column is one parameter point: the modulus ``E(x)`` on top (so the
    nonlinearity being asked for is visible), and below it the deflection --
    reference against PGD -- with the pointwise error on a twin axis.

    Args:
        pgd (CPPGD): the trained decomposition.
        reference (dict, optional): a loaded reference bundle; by default the
            one on disk next to this script.
        labels (list[str], optional): which reference points to draw; defaults
            to the bundle's highlighted ones.
        save_path (str): where to write the PNG.

    Returns:
        dict: the ``relative_errors`` result, computed over *all* the reference
        points, not only the plotted ones.
    """
    import matplotlib.pyplot as plt

    if reference is None:
        reference = load_reference_bundle()
    if labels is None:
        labels = reference["metadata"]["highlight"]

    errors = relative_errors(pgd, reference)
    x_ref = reference["x"]
    index = {label: k for k, label in enumerate(reference["labels"])}

    fig, ax = plt.subplots(
        2, len(labels), figsize=(6.0 * len(labels), 7.0), squeeze=False
    )
    for column, label in enumerate(labels):
        k = index[label]
        params = dict(
            zip(reference["param_names"], (v.item() for v in reference["params"][k]))
        )
        u_ref = reference["u"][k]
        u_pgd = errors["u_pgd"][k]

        top = ax[0][column]
        top.plot(x_ref.numpy(), modulus(x_ref, **params).numpy(), "b-", lw=2)
        top.set_title(
            f"{label}\n"
            + "E1={E1:.0f}, E2={E2:.0f}, alpha={alpha:.1f}, n={n:.1f}".format(**params)
        )
        top.set_xlabel("x")
        top.set_ylabel("E(x)")
        top.grid(True, alpha=0.3)

        bottom = ax[1][column]
        bottom.plot(x_ref.numpy(), u_ref.numpy(), "k-", lw=2, label="FEM reference")
        bottom.plot(x_ref.numpy(), u_pgd.numpy(), "r--", lw=2, label="PGD")
        bottom.set_xlabel("x")
        bottom.set_ylabel("u(x)")
        bottom.grid(True, alpha=0.3)
        bottom.legend(loc="lower left", fontsize="small")
        bottom.set_title(
            f"space-only relative L2 = {errors['per_point'][label]:.2e}",
            fontsize="medium",
        )

        # Pointwise error on a twin axis: a small global L2 can still hide a
        # local failure right at the modulus transition, which is exactly where
        # this problem is hard.
        twin = bottom.twinx()
        twin.plot(x_ref.numpy(), (u_pgd - u_ref).numpy(), color="grey", lw=1, ls=":")
        twin.set_ylabel("PGD - reference", color="grey")
        twin.tick_params(axis="y", colors="grey")

    fig.suptitle(
        f"PGD vs FEM reference -- overall relative L2 "
        f"{errors['overall']:.2e} over all {len(reference['labels'])} reference points"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()
    return errors


def plot_modes(pgd, save_path="pgd5_modes.png"):
    """Plot every mode's factor on every axis, normalised by its max modulus.

    A CP mode is defined up to a per-axis scale (multiply one factor by ``c``,
    divide another by ``c`` and the product is unchanged), so only the *shape*
    of a factor is meaningful -- hence the normalisation. Two modes whose curves
    coincide on every axis are the degenerate case the ``max_correlation``
    diagnostic reports.

    Args:
        pgd (CPPGD): the trained decomposition.
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

    def factor(m, k, pts):
        axis = pgd.axes[k]
        pwi = PointWiseInterpolator(axis.mesh, axis.sf, pgd.monoms[m][k], axis.mapping)
        return pwi.at_position(pts.reshape(-1)).reshape(-1)

    def norm(v):
        peak = v.abs().max()
        return v / peak if peak > 0 else v

    fig, ax = plt.subplots(1, len(AXIS_ORDER), figsize=(4 * len(AXIS_ORDER), 3.5))
    for k, name in enumerate(AXIS_ORDER):
        lo, hi = AXIS_BOUNDS[name]
        pts = torch.linspace(lo, hi, 200)
        for m in range(pgd.n_modes_truncated):
            ax[k].plot(pts.numpy(), norm(factor(m, k, pts)).numpy(), label=f"mode {m}")
        ax[k].set_title(f"{name} factor (normalised)")
        ax[k].set_xlabel(name)
        ax[k].grid(True, alpha=0.3)
    ax[-1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()


def main(verbose=True, train=True, plot=True):
    """Assemble the 5-parametric problem, evaluate the energy once, and train it.

    Always builds the problem and evaluates the energy once at the initial
    (untrained) state -- this proves the whole chain (five axes -> CP-PGD ->
    shared IntegrationDomain -> separated energy) assembles and produces a
    finite value. Unless ``train=False``, it then runs purely greedy PGD
    training (one mode per stage, every earlier mode frozen) and returns the
    ``Problem`` with ``history`` filled.

    Args:
        verbose (bool): print a short summary of the assembled problem and,
            if training, a per-stage diagnostics table.
        train (bool): if True, train the decomposition with ``GreedyTrainer``
            after the initial energy evaluation; if False, return the
            untrained problem (``history`` stays ``None``).
        plot (bool): if True (and training ran), draw the convergence curve, the
            PGD-vs-reference comparison and the per-axis mode factors. Requires
            matplotlib, which is imported lazily inside the plotting helpers.

    Returns:
        Problem: the assembled objects, for interactive use.
    """
    problem = build_problem(lambda layout, pgd: energy(layout, pgd, load_name="load"))

    field_layout = problem.model()
    value = problem.model.loss(field_layout)

    if verbose:
        n_nodes_actual = {
            name: problem.axes[name].topology.n_nodes for name in AXIS_ORDER
        }
        print("axes            :", [axis.name for axis in problem.pgd.axes])
        print("nodes per axis  :", n_nodes_actual)
        print("active modes    :", problem.pgd.n_modes_truncated)
        print(f"energy          : {value.item():.6e}")

    if not train:
        return problem

    # Purely greedy: one mode per stage, every earlier mode frozen for good.
    #
    # min_iter is load-bearing twice over. Adam spends a long sticky early phase
    # on this energy where the loss barely moves, which a plateau detector reads
    # as convergence. And below ~80 iterations per stage the greedy step simply
    # rediscovers mode 0 -- max_correlation reads 1.0 and the "modes" are copies
    # of each other (see CHANGELOG). The printed max corr column is what tells
    # you whether that is happening.
    trainer = GreedyTrainer(
        problem.model,
        stage_criterion=RelativeChange(tol=1e-3, window=20, max_iter=600, min_iter=120),
        enrichment_criterion=RelativeGain(tol=1e-3),
    )
    history = trainer.enrich()
    problem.history = history

    if verbose:
        print()
        print(f"training stopped: {history.stop_reason}")
        print(
            f"{'stage':>5} {'iters':>6} {'stop':>10} "
            f"{'energy':>14} {'gain':>12} {'amplitude':>11} {'max corr':>9}"
        )
        for record in history.stages:
            print(
                f"{record.stage:5d} {record.n_iter:6d} {record.stop_reason:>10} "
                f"{record.energy:14.6e} {record.gain:12.4e} "
                f"{record.diagnostics['amplitude']:11.4e} "
                f"{record.diagnostics['max_correlation']:9.3f}"
            )

    if plot:
        plot_convergence(history)
        errors = plot_solution(problem.pgd)
        plot_modes(problem.pgd)
        if verbose:
            per_point = errors["per_point"]
            print()
            print("relative L2 error vs the FEM reference")
            print(
                f"  overall (space and parameter points jointly): {errors['overall']:.3e}"
            )
            print(f"  worst single point ...........: {max(per_point.values()):.3e}")
            for label, value in per_point.items():
                print(f"    {label:>17} (space only): {value:.3e}")

    return problem


if __name__ == "__main__":
    main()
