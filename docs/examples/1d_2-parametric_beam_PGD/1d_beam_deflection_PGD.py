"""Parametric 1D bar deflection via CP-PGD -- runnable example.

Solves u(x, E) for a bi-clamped bar under a constant axial load, with the
Young's modulus E treated as an extra (parametric) coordinate. The solution is
sought in separated form u(x, E) = sum_m w_m^x(x) * w_m^E(E) and built greedily,
one mode at a time. See the accompanying ``1d_beam_deflection_PGD.md`` for the
maths. Run directly to train and produce the two figures:

    python 1d_beam_deflection_PGD.py
"""

from dataclasses import dataclass

import torch

from neurom.decompositions import Axis, CPPGD
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import MidPoint1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.interpolation.integration_domain import IntegrationDomain
from neurom.constraints import Dirichlet, NoConstraint
from neurom.differential import jacobian_field
from neurom.inner import inner
from neurom.integrate import integrate
from neurom.field_layout import FieldLayout
from neurom.training import GreedyTrainer, RelativeChange, RelativeGain

torch.set_default_dtype(torch.float32)

# --- domain and parameter intervals ---------------------------------------
X_MIN, X_MAX = 0.0, 10.0
E_MIN, E_MAX = 10.0, 100.0
DEFAULT_N_NODES = {"space": 30, "E": 20}
LOAD_VALUE = 1000.0
N_MODES_MAX = 3


def DEFAULT_STAGE_CRITERION():
    """Greedy stage criterion calibrated for this problem.

    A factory, not a constant, so each caller gets a fresh criterion.
    ``min_iter`` clears Adam's sticky early phase on this energy; the
    denominator floor inside ``RelativeChange`` handles its zero crossing.
    """
    return RelativeChange(tol=1e-3, window=20, max_iter=600, min_iter=120)


def DEFAULT_ENRICHMENT_CRITERION():
    """Greedy enrichment criterion calibrated for this problem."""
    return RelativeGain(tol=1e-3)


# Calibrated in Step 6 against an actual run (deterministic -- no randomness in
# this pipeline beyond the deterministic 0.5*ones seed, confirmed identical
# across manual_seed(0) and manual_seed(42)): measured relative L2 error
# 0.1019, roughly doubled and rounded up.
#
# This is NOT a comparison against final_error_tol / strict_error_tol in
# tests/integration/test_1d_beam_deflection_PGD.py -- those are tolerances,
# not measurements, and an earlier version of this comment wrongly compared
# against them. The hand-rolled loop's actual greedy-regime error (rerun with
# -s) is 0.0954, so GreedyTrainer's 0.1019 is marginally worse (~7%
# relative), not better; its joint-polish error is 0.0153, not 3%, and that
# extra all-modes stage is not something GreedyTrainer performs. See
# CHANGELOG.md (2026-07-24) for the full comparison.
ANALYTICAL_ERROR_TOL = 0.21


def make_axis(name, lo, hi, n_nodes, constraint, sf, quad, mapping, init_value=0.5):
    """Build one Axis on a uniform 1-D mesh of ``n_nodes`` nodes over [lo, hi].

    Wraps the Topology / Field / Axis boilerplate so the two axes are not two
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
    """Everything the 2-parametric beam problem needs to train and be checked.

    Attributes:
        model (NeuROMModel): The model, with the energy injected as its loss.
        pgd (CPPGD): The separated representation.
        field_layout (FieldLayout): Holds every field, including the load.
        domain (IntegrationDomain): Interpolates all active fields.
        axes (dict[str, Axis]): The axes, keyed by name.
        x_min, x_max, E_min, E_max (float): Axis bounds. Carried on the problem
            because this is the one case whose exact analytical solution can be
            evaluated from them.
        load_value (float): Constant load q, likewise needed by that solution.
        history (TrainingHistory | None): Filled by ``main`` when it trains.
    """

    model: object
    pgd: object
    field_layout: object
    domain: object
    axes: dict
    x_min: float
    x_max: float
    E_min: float
    E_max: float
    load_value: float
    history: object = None


def build_problem(
    loss_fn, *, n_modes_max=N_MODES_MAX, n_modes_ini=1, n_nodes=None, quad=None
):
    """Assemble the space and E axes, the CP-PGD, the load and the model.

    The energy is *injected*: this function never references ``energy``
    directly, so the same wiring drives a different (e.g. non-linear PGD)
    functional unchanged.

    Args:
        loss_fn (Callable): ``loss_fn(field_layout, decomposition, load_name) ->
            Tensor``.
        n_modes_max (int): Mode budget of the CP-PGD.
        n_modes_ini (int): Number of initially active (trainable) modes.
        n_nodes (dict[str, int], optional): Per-axis node counts overriding
            ``DEFAULT_N_NODES``; used by the tests to build a tiny problem.
        quad (QuadratureRule, optional): Shared quadrature rule for every axis;
            defaults to ``MidPoint1D()`` (one point per element).

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
    axis_space = make_axis(
        "space",
        X_MIN,
        X_MAX,
        n_x,
        Dirichlet(nodes=[0, n_x - 1], values_imposed=torch.zeros(2, 1)),
        sf,
        quad,
        mapping,
    )
    axis_E = make_axis(
        "E", E_MIN, E_MAX, counts["E"], NoConstraint(), sf, quad, mapping
    )
    axes = [axis_space, axis_E]

    pgd = CPPGD(axes=axes, n_modes_max=n_modes_max, name="pgd", n_modes_ini=n_modes_ini)

    ###### Define constant load.
    # The load is a *field* f(x), not a raw nodal vector: it has to be
    # sampled at the SAME quadrature points as u so that inner(f, u) aligns
    # (this is exactly what neurom.physics.LoadPotential does). We give it its
    # own nodal values on the space mesh, then interpolate it once on the
    # space axis's quadrature (same sf / quad / mapping / topology as u).
    # For a load that is constant in E, this is the single rank-1 spatial
    # factor f_0(x) of the separated source f(x, E) = f_0(x) ⊗ 1(E); the E
    # factor "1" is what the Gm = ∫ lmbda dE term below carries implicitly.
    load_field = field_layout.add(
        Field(
            name="load",
            topology=axis_space.topology,
            values=LOAD_VALUE * torch.ones(n_x, 1),
        )
    )
    assembly_f = QuadratureAssembly(axis_space.context, sf, load_field)

    domain = IntegrationDomain([*pgd.assemblies(), assembly_f])  # do not forget *

    model = NeuROMModel(
        field_layout=field_layout,
        decomposition=pgd,
        integration_domain=domain,
        loss=lambda layout: loss_fn(layout, pgd, load_name="load"),
    )

    return Problem(
        model=model,
        pgd=pgd,
        field_layout=field_layout,
        domain=domain,
        axes={axis.name: axis for axis in axes},
        x_min=X_MIN,
        x_max=X_MAX,
        E_min=E_MIN,
        E_max=E_MAX,
        load_value=LOAD_VALUE,
    )


def main():
    """Build the 2-parametric beam, train it greedily, and plot the result.

    Stage length is no longer a fixed iteration count -- it is decided by
    ``DEFAULT_STAGE_CRITERION``, and enrichment by
    ``DEFAULT_ENRICHMENT_CRITERION``. Pass different criteria to
    :class:`~neurom.training.GreedyTrainer` to change that.

    Returns:
        Problem: The assembled problem, with ``history`` filled by training.
    """
    problem = build_problem(energy)

    trainer = GreedyTrainer(
        problem.model,
        stage_criterion=DEFAULT_STAGE_CRITERION(),
        enrichment_criterion=DEFAULT_ENRICHMENT_CRITERION(),
    )
    history = trainer.enrich()
    problem.history = history
    print(f"Successfully trained! ({history.stop_reason})")

    ## Plotting
    plot_convergence(history.losses)  # OK
    plot_solution(  # investigate how to get the information monom per monom
        problem.model,
        problem.pgd,
        x_min=problem.x_min,
        x_max=problem.x_max,
        E_min=problem.E_min,
        E_max=problem.E_max,
        load_value=problem.load_value,
    )


## Plotting helpers
def plot_convergence(loss_history, save_path="pgd_convergence.png"):
    """Plot the (minimised) energy against the training iteration.

    Args:
        loss_history (list[float]): energy value returned by the optimizer at each
            iteration (the PGD functional we minimise).
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(loss_history)), loss_history, "b-")
    ax.set_title("Training convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy (loss)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()


def plot_solution(
    model,
    pgd_approx,
    *,
    x_min,
    x_max,
    E_min,
    E_max,
    load_value,
    save_path="pgd_vs_analytical.png",
):
    """Compare the PGD solution to the analytical beam deflection.

    Four panels: (1) the full solution u(x, E) at a fixed E, PGD vs analytical;
    (2) same at a fixed x, swept over E; (3) the space factor of every mode;
    (4) the E factor of every mode. See the per-panel comments for the scaling
    caveat on the mode-by-mode factors.

    Args:
        model (NeuROMModel): trained model (put in eval mode here).
        pgd_approx (CPPGD): the decomposition, to read per-mode / per-axis factors.
        x_min, x_max, E_min, E_max (float): axis bounds.
        load_value (float): constant load q used in the analytical formula.
        save_path (str): where to write the PNG.
    """
    import matplotlib.pyplot as plt

    model.eval()

    # --- helper: evaluate a single factor w_m^k(.) at arbitrary points ------
    # A PGD mode is u_m(x, E) = w_m^space(x) * w_m^E(E). Each factor is one
    # monom field of the decomposition; PointWiseInterpolator evaluates it on
    # its own axis mesh at arbitrary query points (same tool the decomposition
    # uses internally in evaluate()/assemble()).
    def factor(m, k, pts):
        pwi = PointWiseInterpolator(
            pgd_approx.axes[k].mesh,
            pgd_approx.axes[k].sf,
            pgd_approx.monoms[m][k],
            pgd_approx.axes[k].mapping,
        )
        return pwi.at_position(pts.reshape(-1)).reshape(-1)

    def norm(v):
        m = v.abs().max()
        return v / m if m > 0 else v

    n_modes = pgd_approx.n_modes_truncated

    # --- 1) FULL solution: PGD (sum of modes) vs analytical, at fixed E -----
    # This is the unambiguous comparison. The analytical deflection is
    #   u(x, E) = 0.5 q (x - x_min)(x - x_max) / E,
    # which is itself rank-1 separable: a space parabola times 1/E.
    E_fixed = E_max / 2
    x_plot = torch.linspace(x_min, x_max, 200)
    E_plot = E_fixed * torch.ones_like(x_plot)
    u_pgd_full = model(torch.stack([x_plot, E_plot], dim=1)).reshape(
        -1
    )  # sum_m w_m^x w_m^E
    u_ana_full = 0.5 * load_value * (x_plot - x_min) * (x_plot - x_max) / E_plot

    # Counterpart of panel 1: fix x at mid-span, sweep E. Same diagonal
    # evaluate (each E paired with the same x_fixed), so we see the parametric
    # dependence u(x_fixed, .) -- a 1/E curve -- rather than the spatial parabola.
    x_fixed = (x_min + x_max) / 2
    E_sweep = torch.linspace(E_min, E_max, 200)
    x_sweep = x_fixed * torch.ones_like(E_sweep)
    u_pgd_E = model(torch.stack([x_sweep, E_sweep], dim=1)).reshape(-1)
    u_ana_E = 0.5 * load_value * (x_fixed - x_min) * (x_fixed - x_max) / E_sweep

    # --- 2) MODE per MODE, AXIS per AXIS -----------------------------------
    # Caveat: individual factors carry an arbitrary scale (w_m^x -> c w_m^x,
    # w_m^E -> w_m^E / c leaves the product unchanged), so only their SHAPE is
    # comparable per axis, not the amplitude. We therefore normalise each
    # curve by its max |.|. The analytical solution is rank-1, so its space
    # shape is (x-x_min)(x-x_max) and its E shape is 1/E -- overlaid as the
    # reference each PGD mode is trying to capture (mode 0 should match it,
    # higher modes should be ~0 since the truth has a single mode).
    x_fac = torch.linspace(x_min, x_max, 200)
    E_fac = torch.linspace(E_min, E_max, 200)
    ana_x_shape = (x_fac - x_min) * (x_fac - x_max)
    ana_E_shape = 1.0 / E_fac

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))

    # Full pgd vs analytical at fixed E (sweep x)
    ax[0].plot(x_plot.numpy(), u_ana_full.numpy(), "k-", lw=2, label="analytical")
    ax[0].plot(
        x_plot.numpy(),
        u_pgd_full.detach().numpy(),
        "r--",
        lw=2,
        label="PGD (sum of modes)",
    )
    ax[0].set_title(f"Full solution u(x, E={E_fixed:.0f})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("u")
    ax[0].legend()

    # Full pgd vs analytical at fixed x (sweep E) -- counterpart of panel 0
    ax[1].plot(E_sweep.numpy(), u_ana_E.numpy(), "k-", lw=2, label="analytical")
    ax[1].plot(
        E_sweep.numpy(),
        u_pgd_E.detach().numpy(),
        "r--",
        lw=2,
        label="PGD (sum of modes)",
    )
    ax[1].set_title(f"Full solution u(x={x_fixed:.0f}, E)")
    ax[1].set_xlabel("E")
    ax[1].set_ylabel("u")
    ax[1].legend()

    # approx space (per mode)
    for m in range(n_modes):
        ax[2].plot(
            x_fac.numpy(),
            norm(factor(m, 0, x_fac)).detach().numpy(),
            label=f"PGD mode {m}",
        )

    # analytical space
    ax[2].plot(
        x_fac.numpy(), norm(ana_x_shape).numpy(), "k:", lw=2, label="analytical shape"
    )
    ax[2].set_title("Space factor w_m^x(x)  (normalised shape)")
    ax[2].set_xlabel("x")
    ax[2].legend()

    # approx E (per mode)
    for m in range(n_modes):
        ax[3].plot(
            E_fac.numpy(),
            norm(factor(m, 1, E_fac)).detach().numpy(),
            label=f"PGD mode {m}",
        )

    # analytical E
    ax[3].plot(
        E_fac.numpy(),
        norm(ana_E_shape).numpy(),
        "k:",
        lw=2,
        label="analytical 1/E shape",
    )
    ax[3].set_title("E factor w_m^E(E)  (normalised shape)")
    ax[3].set_xlabel("E")
    ax[3].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.show()


## Energy


def energy(field_layout: FieldLayout, decomposition: any, load_name: str):
    # on va chercher les noms des champs {'space': ['pgd_dimspace_mode0'], 'E': ['pgd_dimE_mode0']}
    directory = decomposition.directory()
    n_modes = len(directory["space"])

    # on les récupère dans le field_layout
    space_modes_names = directory["space"]
    E_modes_names = directory["E"]
    space_modes = [field_layout[name] for name in space_modes_names]
    E_modes = [field_layout[name] for name in E_modes_names]

    # et le load
    load_field = field_layout[load_name]

    ## Elastic
    elastic = 0.0
    # param part
    E_val = [E_mode_field.x for E_mode_field in E_modes]
    lmbdas = [E_mode_field.u for E_mode_field in E_modes]
    J_E = [E_mode_field.measure for E_mode_field in E_modes]

    # space part
    u = [space_mode_field.u for space_mode_field in space_modes]
    x_val = [space_mode_field.x for space_mode_field in space_modes]
    # jacobian_field returns (N_e, N_q, *u_shape, d): one extra trailing axis of
    # size d (the physical dimension) compared to u's own shape (N_e, N_q,
    # *u_shape). That axis must be *contracted away* -- the elastic term is the
    # scalar product grad(u_m) . grad(u_n) summed over the spatial directions --
    # so we keep the raw jacobian_field output here (no reshape) and let inner()
    # perform the contraction in the loop below. See Kx there.
    #
    # This replaces an earlier `.reshape(u[n].shape)` on the line below. That
    # reshape only appeared to work in 1D: with d=1 the extra axis has size 1 and
    # reshape could drop it, but it left `grad_u[m] * grad_u[n] * J_u[m]` as a
    # plain element-wise product -- a broadcasting trap. With MidPoint1D (N_q=1)
    # that silently built an (N_e, N_e, 1, 1) outer product over *elements*
    # (numerically wrong energy, no exception); with N_q>1 (e.g. TwoPoints1D) the
    # shapes don't broadcast and it hard-failed with "size of tensor a (2) must
    # match size of tensor b (N_e)"; and in 2D/3D (d>1) reshape can't collapse the
    # axis at all. Using inner() instead is correct AND dimension-agnostic.
    grad_u = [jacobian_field(x=x_val[n], u=u[n]) for n in range(n_modes)]
    J_u = [space_mode_field.measure for space_mode_field in space_modes]

    # NB: cross terms (m, n) below assume modes m and n share the same mesh
    # (measure/coords indexed by m are used for both). Once modes can live on
    # independent (e.g. r-adapted) meshes, these products need to be
    # integrated on a common intersection mesh with a recomputed measure.
    for m in range(n_modes):
        for n in range(n_modes):
            # inner() contracts grad(u_m) . grad(u_n) over the field and d axes,
            # returning (N_e, N_q, 1) -- same rank as the measure J_u -- so the
            # `* J_u[m]` below aligns element-wise as intended (no reshape needed).
            Kx = integrate(inner(grad_u[m], grad_u[n]) * J_u[m])
            AE = integrate(E_val[m] * lmbdas[m] * lmbdas[n] * J_E[m])
            elastic = elastic + Kx * AE
    elastic = 0.5 * elastic
    load = 0.0
    # load_interp.u is the load field sampled at the space quadrature points, so
    # it has the same (N_e, N_q, *u_shape) shape as u[m]: inner() contracts them
    # into (N_e, N_q, 1) and the * J_u[m] measure aligns element-wise -- unlike
    # the old raw nodal `external_load_values` (N_space, 1), which broadcast wrong
    # against quadrature-point values (silently with N_q=1, crashing with N_q>1).
    # Gm = ∫ lmbda dE carries the constant-in-E factor of the separated load.
    load_f = load_field.u
    for m in range(n_modes):
        Fx = integrate(inner(load_f, u[m]) * J_u[m])
        Gm = integrate(lmbdas[m] * J_E[m])
        load = load + Fx * Gm

    return elastic + load


if __name__ == "__main__":
    main()
