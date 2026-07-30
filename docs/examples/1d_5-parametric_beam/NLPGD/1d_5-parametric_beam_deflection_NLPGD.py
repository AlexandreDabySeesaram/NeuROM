"""Five-parametric 1D bar with a tanh-graded Young's modulus -- non-linear PGD.

The polynomial-NL-PGD twin of the sibling ``PGD/`` example. Same bar, same load,
same five axes, same FEM reference; the *only* thing that changes is the format
of the separated representation, so the two directories can be scored against
each other point for point.

Bi-clamped bar under a constant axial load, whose modulus has two zones E1 and E2
separated by a tanh transition centred at ``alpha`` with slope ``n``:

    E(x, E1, E2, alpha, n) = (E2 - E1) / 2 * tanh(n (x - alpha)) + (E2 + E1) / 2

The four material/geometry parameters are treated as extra coordinates. Where
the CP example seeks

    u = sum_i X_i(x) lambda_i(E1) mu_i(E2) A_i(alpha) N_i(n),

this one seeks the polynomial extension of
:class:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD`

    u = sum_i ( prod_j w_ij + sum_{lambda in I} C_{i lambda} prod_j w_ij^lambda_j )

with the exponent set ``I`` chosen by :func:`build_exponents` (a ``RunConfig``
knob, so the sweep can vary it). Every term is still a product over axes, so the
energy still factorises into 1-D moments -- but the unit of summation is now the
**term**, not the mode, and the bilinear double loop is quadratic in
``n_modes * (1 + |I|)``. That cost is why ``N_MODES_MAX`` is smaller here than in
the CP example; see the note on :func:`energy`.

Because ``E1, E2 > 0`` and ``tanh`` maps into (-1, 1), the modulus stays strictly
between E1 and E2 -- positivity is structural, no clamping needed.

This module builds the problem (axes, decomposition, energy, model), evaluates
the energy once, and then trains it in ``main``, reporting a per-stage
diagnostics table. Pass ``train=False`` for assembly only, ``plot=False`` to skip
the figures.

Six training strategies are selectable, all scored against the same reference::

    python .../1d_5-parametric_beam_deflection_NLPGD.py joint
    python .../1d_5-parametric_beam_deflection_NLPGD.py staged
    python .../1d_5-parametric_beam_deflection_NLPGD.py refine
    python .../1d_5-parametric_beam_deflection_NLPGD.py support
    python .../1d_5-parametric_beam_deflection_NLPGD.py greedy
    python .../1d_5-parametric_beam_deflection_NLPGD.py simultaneous

All of them are purely greedy in the modes: a finished mode is frozen for good,
its coefficient row included. They differ in how one mode is fitted before that
freeze (see :data:`SCHEDULES`):

===============  ===========================================================
``joint``        one stage: the monoms and ``C`` descend together throughout
``staged``       two stages: CP mode first, then ``C`` alone, monoms frozen
``refine``       two stages: CP mode first, then monoms and ``C`` together
``support``      two stages: CP mode first, then the **space monom frozen**
                 as a support, parametric monoms and ``C`` fitted over it
``greedy``       CP baseline -- ``C`` never released
``simultaneous`` CP baseline, every mode retrained at each enrichment
===============  ===========================================================

``joint`` and ``staged`` are the pair this example was first built to compare.
The two library trainers never call ``unfreeze_mode_coefficients``, so ``C``
stays at its zero-initialisation and their runs are by construction
**bit-identical to CP-PGD** -- that is the point: they are the controlled
baseline, run through exactly the same energy and the same reference.

Two further knobs cut across the schedules, both off by default and both
``RunConfig`` fields rather than strategies, so they compose with any of them:

``n_linear_modes``
    How many leading modes are trained as plain CP before the schedule starts.
    The non-linearity then never acts from a mode's seed -- it corrects a linear
    decomposition that already exists.
``leading_coefficient``
    Gives each non-linear mode a trainable weight ``c_i`` on its own linear
    term, so a late mode may carry almost no linear part at all. It also
    switches the gauge fix to normalise every axis, which is what puts ``c_i``
    and every row of ``C_i`` on one scale.

``n_linear_modes`` with ``support`` is the combination the plan settles on: a
few linear modes, then frozen-support corrections.

Each writes its trained model to ``param_sweep/nlpgd5_<strategy>.pt`` and
**reuses it on the next run**: re-running the command above redraws the figures
from the checkpoint in seconds. Add ``--retrain`` to train again and overwrite
it. Training prints a per-stage progress bar; a stage is minutes long, so a
silent run would be indistinguishable from a hung one. Figures are written to
``plots/``.

Accuracy is judged against ``../reference/reference_fem_solution.py`` -- the
reference is **shared with the CP example**, one directory up, which is what
makes the comparison honest. It is a direct, non-reduced FEM solve at a handful
of parameter points, saved to ``../reference/reference_solution.pt``. Generate it
once before plotting::

    python docs/examples/1d_5-parametric_beam/reference/reference_fem_solution.py
"""

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from neurom.constraints import Dirichlet, NoConstraint
from neurom.decompositions import (
    Axis,
    PolynomialNLPGD,
    pin_axis,
    total_degree_exponents,
    uniform_exponents,
)
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
from neurom.training import (
    GreedyTrainer,
    ProgressBar,
    RelativeChange,
    RelativeGain,
    SimultaneousTrainer,
    load_checkpoint,
    save_checkpoint,
)

torch.set_default_dtype(torch.float32)

#: Where trained models and figures live, one directory each so the top level of
#: the example stays to scripts and docs. The FEM reference is deliberately NOT
#: here: it sits one level up, shared with the CP example, so both formats are
#: scored against the very same numbers.
HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "plots"
PARAM_SWEEP_DIR = HERE / "param_sweep"
REFERENCE_DIR = HERE.parent / "reference"
PLOT_DIR.mkdir(exist_ok=True)
PARAM_SWEEP_DIR.mkdir(exist_ok=True)

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
# Half the CP example's budget, and deliberately so: `energy` loops over PAIRS OF
# TERMS, and a mode now carries `1 + |I|` terms instead of 1. At the default
# I = uniform_exponents(5, 3) that is 3 terms per mode, so 5 modes already cost
# 225 pairs where CP-PGD's 10 modes cost 100.
#
# Measured on DEFAULT_N_NODES, forward + backward, every coefficient row
# released (macOS/CPU, float32):
#
#     modes   terms   pairs   ms/iteration
#         1       3       9              6
#         3       9      81             42
#         5      15     225            110
#
# So a 5-mode run of ~600-iteration stages is minutes, not hours. Raising this
# is affordable; raising `max_power` is the expensive direction, since the pair
# count goes as the SQUARE of (1 + |I|).
N_MODES_MAX = 5

#: Default exponent set: the plan's "simple case" at max_power = 3, i.e.
#: I = {(2,2,2,2,2), (3,3,3,3,3)} -- two correction terms per mode. Overridable
#: per run through ``RunConfig.exponent_set`` / ``RunConfig.max_power``.
DEFAULT_EXPONENT_SET = "uniform"
DEFAULT_MAX_POWER = 3

# Axis order is load-bearing: it fixes the column order of the
# decomposition's evaluate(), the key order of directory() and the per-axis
# exponent order of polynomial_directory().
AXIS_ORDER = ["space", "E1", "E2", "alpha", "n"]

AXIS_BOUNDS = {
    "space": (X_MIN, X_MAX),
    "E1": (E1_MIN, E1_MAX),
    "E2": (E2_MIN, E2_MAX),
    "alpha": (ALPHA_MIN, ALPHA_MAX),
    "n": (N_MIN, N_MAX),
}


def make_axis(
    name, lo, hi, n_nodes, constraint, sf, quad, mapping, init_value, init_value_rest=None
):
    """Build one Axis on a uniform 1-D mesh of ``n_nodes`` nodes over [lo, hi].

    Wraps the Topology / Field / Axis boilerplate so the five axes are not five
    copy-pasted blocks. The Axis builds its own Mesh and QuadratureContext.

    Args:
        name (str): Axis name; also the prefix of its nodes-positions field.
        lo, hi (float): Interval bounds.
        n_nodes (int): Number of mesh nodes (so ``n_nodes - 1`` linear elements).
        constraint (Constraint): Dirichlet or NoConstraint for the monoms.
        sf (ShapeFunction), quad (QuadratureRule), mapping: shared discretisation.
        init_value (float): Constant seed for mode 0's nodal values (and every
            mode, if ``init_value_rest`` is left ``None``).
        init_value_rest (float, optional): Constant seed for every mode after
            the first. Defaults to ``None``, which reuses ``init_value`` for
            all modes.

    Returns:
        Axis: ready to be handed to the decomposition.
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
        init_values_rest=(
            init_value_rest * torch.ones(n_nodes, 1)
            if init_value_rest is not None
            else None
        ),
    )


@dataclass(frozen=True)
class RunConfig:
    """One training-hyperparameter configuration for the sweep.

    Every knob explicit and flat. ``name`` is a human label only; identity is
    the content hash (see :func:`config_id`), so renaming does not change which
    ledger row / checkpoint a config maps to. ``n_nodes=None`` means
    ``DEFAULT_N_NODES``.
    """

    name: str
    strategy: str = "joint"
    stage_tol: float = 1e-5
    window: int = 20
    max_iter: int = 600
    min_iter: int = 120
    stage_floor: float = 1.0
    enrichment_tol: float = 1e-5
    enrichment_floor: float = 1.0
    lr: float = 0.1
    #: Adam learning rate for the coefficient rows, kept separate from ``lr``
    #: because the two parameter families live on wildly different scales -- see
    #: :func:`build_optimizer_factory` for the measurement. Sharing ``lr``
    #: diverges the ``staged`` schedule outright. The default is the best of the
    #: five values tried there, on a tiny mesh; it is a starting point for a
    #: sweep, not a tuned value.
    coefficient_lr: float = 1e-3
    n_modes_max: int = 5
    n_nodes: dict = None
    #: Which exponent set ``I`` the polynomial modes carry -- ``"uniform"`` for
    #: :func:`~neurom.decompositions.uniform_exponents` (every axis at the same
    #: power) or ``"total_degree"`` for
    #: :func:`~neurom.decompositions.total_degree_exponents` (anisotropic, all
    #: exponents >= 1 with bounded sum). A string, not a tensor, because
    #: :func:`config_id` hashes the config as JSON.
    exponent_set: str = DEFAULT_EXPONENT_SET
    #: The bound that set is built with: the largest power for ``"uniform"``,
    #: the largest total degree for ``"total_degree"`` (which needs
    #: ``max_power > 5`` here, five being the number of axes).
    max_power: int = DEFAULT_MAX_POWER
    #: Amplitude/shape seed for every mode after the first (see
    #: :func:`build_problem`). A fresh mode enters with its parametric factors at
    #: unit shape and the whole product's small initial amplitude carried by the
    #: space factor, so enrichment does not spike the loss the way a 0.5 seed on
    #: every factor does. Must stay > 0: an all-zero factor is a stationary point
    #: of the energy and the mode never takes off (see ``CPPGD.add_mode``).
    seed_amplitude: float = 0.05
    #: How many leading modes are trained as **pure CP**, one stage each, before
    #: the polynomial schedule starts. The "curve a space that a linear PGD has
    #: already laid out" idea: the non-linearity does not act directly, it
    #: corrects a linear decomposition that already exists. ``0`` (the default)
    #: applies the schedule from mode 0 and reproduces the historical behaviour.
    #:
    #: A fixed count rather than a coarse stagnation tolerance, deliberately: it
    #: keeps the rank split inside ``config_id``, so two rows at the same ``l``
    #: are comparable. ``linear_stage_tol`` supplies the "coarse" half.
    n_linear_modes: int = 0
    #: Release each non-linear mode's leading coefficient ``c_i``, so a mode may
    #: carry little or no linear part at all. Also switches ``renormalise`` to
    #: the all-``d``-axes gauge fix, which is what puts ``c_i`` and every row of
    #: ``C_i`` on one scale -- see
    #: :meth:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD.renormalise_mode`.
    #: Only the modes at or after ``n_linear_modes`` get one released; the linear
    #: phase stays pure CP.
    leading_coefficient: bool = False
    #: Pin the **space** exponent to 1 via
    #: :func:`~neurom.decompositions.pin_axis`, so the correction terms are
    #: non-linear in the parameters only and the space factor stays a plain
    #: support. The natural companion of the ``support`` schedule, but
    #: independent of it so the sweep can separate the two.
    pin_space_exponent: bool = False
    #: Make every correction term **orthogonal to its mode's leading term**, by
    #: deflating one axis' factor:
    #: ``w_j^p -> w_j^p - (<w_j^p, w_j>/<w_j, w_j>) w_j``.
    #:
    #: Aimed at a different failure from ``renormalise``, and the two are
    #: independent. ``renormalise`` removes the *leverage* that let the
    #: highest-degree row move the field fastest per Adam step -- it stopped
    #: ``uniform3`` diverging (209% -> 8.8% overall error) but left the
    #: amplitudes where they were: measured on ``944186d3``, the ``(3,3,3,3,3)``
    #: row still holds 99.5% of mode 3 and 100% of mode 5.
    #:
    #: What is left is *redundancy*. With the monoms free, ``prod_j w_j^p``
    #: spans exactly the same rank-1 set as ``prod_j w_j`` (take
    #: ``w_j -> w_j^(1/p)``), so a lone correction can **replace** the leading
    #: term rather than complement it -- and a mode with 99% in one term is CP
    #: again, re-expressed with a tiny coefficient against a huge basis. Nothing
    #: in the energy distinguishes the two, so nothing pushes back. Deflation
    #: removes the overlap outright, and it is *exact*: no penalty, no
    #: hyperparameter, no bias on the minimiser.
    #:
    #: Costs one extra term row per correction (deflation is linear, so the
    #: deflated product expands into two ordinary ones), and the energy's double
    #: loop is quadratic in the term count: ``1 + 2|I|`` against ``1 + |I|``, so
    #: 2.8x at ``|I| = 2``.
    #:
    #: **Only leading-vs-correction overlap is removed.** Two corrections still
    #: overlap each other; ``uniform3``'s ``(2,2,2,2,2)`` vs ``(3,3,3,3,3)``
    #: contest is untouched, and whether it matters is unmeasured.
    orthogonal_corrections: bool = False
    #: Stage tolerance during the linear phase only -- the "tol de stagnation
    #: coarse". ``None`` reuses ``stage_tol``, so the phase split costs nothing
    #: unless it is asked for.
    linear_stage_tol: float = None
    #: Fix the scale gauge at the start of every stage
    #: (:meth:`~neurom.training.base.PGDTrainer.fix_gauge`). Exposed as a knob so
    #: the gauge fix can be *measured* rather than assumed: it is the default
    #: because the decomposition has flat directions without it, but nothing in
    #: the ledger has yet compared a run with and without.
    #:
    #: The invariance is ``w_ij -> s_j w_ij`` with the coefficients absorbing the
    #: factor. Writing ``L`` for the exponent rows whose coefficient is *fixed*,
    #: the flat directions per mode number ``d - rank(L)``: ``d - 1`` here
    #: (``L`` is the leading row alone), or ``d`` with ``leading_coefficient``,
    #: which is why the two knobs are coupled --
    #: :meth:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD.renormalise_mode`
    #: imposes exactly as many conditions as there are directions.
    #:
    #: Turning it **off with** ``leading_coefficient=True`` leaves the full
    #: ``d``-dimensional orbit unfixed -- more degeneracy than the historical
    #: default, not less. That combination is a deliberate control, not a
    #: recommendation.
    #:
    #: A no-op for the CP baselines: ``CPPGD.renormalise()`` does nothing, so
    #: ``greedy``/``simultaneous`` rows are unaffected either way.
    renormalise: bool = True


def config_id(cfg):
    """Stable 8-hex-char identity of a config's *contents*.

    Hash over every field with sorted keys, so field order is irrelevant and
    any knob change yields a new id (hence a new ledger row and checkpoint).
    """
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


#: Exponent-set builders, keyed by ``RunConfig.exponent_set``. Injected the same
#: way ``STRATEGIES`` is, so adding a third set is one entry here and nothing else.
EXPONENT_SETS = {
    "uniform": uniform_exponents,
    "total_degree": total_degree_exponents,
}


def build_exponents(cfg, n_axes=len(AXIS_ORDER)):
    """Turn a ``RunConfig``'s ``(exponent_set, max_power)`` into the tensor ``I``.

    Both builders reject a bound that makes the set empty or degenerate, naming
    it in the message -- ``"uniform"`` needs ``max_power >= 2`` (power 1 is the
    leading term, carried separately with a fixed unit coefficient) and
    ``"total_degree"`` needs ``max_power > n_axes``, so ``>= 6`` for this
    example's five axes.

    With ``cfg.pin_space_exponent`` the result is passed through
    :func:`~neurom.decompositions.pin_axis` on axis 0, which is ``space`` (see
    ``AXIS_ORDER``): the corrections then curve the *parameter* directions only
    and leave the space factor entering linearly, as a support. Pinning can
    shrink the set -- ``pin_axis`` deduplicates and drops the leading term -- so
    ``|I|`` is not a function of ``max_power`` alone once it is on.

    Args:
        cfg (RunConfig): supplies ``exponent_set``, ``max_power`` and
            ``pin_space_exponent``.
        n_axes (int): number of axes; defaults to the five of this problem.

    Returns:
        torch.Tensor: ``(T, n_axes)`` long tensor, ready for ``PolynomialNLPGD``.
    """
    try:
        builder = EXPONENT_SETS[cfg.exponent_set]
    except KeyError:
        raise ValueError(
            f"unknown exponent_set {cfg.exponent_set!r}; "
            f"pick one of {sorted(EXPONENT_SETS)}"
        ) from None
    exponents = builder(n_axes, cfg.max_power)
    if cfg.pin_space_exponent:
        exponents = pin_axis(exponents, axis=AXIS_ORDER.index("space"), power=1)
    return exponents


def build_criteria(cfg):
    """Turn a ``RunConfig`` into its (stage, enrichment) criteria."""
    stage = _stage_criterion(cfg, cfg.stage_tol)
    enrichment = RelativeGain(tol=cfg.enrichment_tol, floor=cfg.enrichment_floor)
    return stage, enrichment


def build_linear_stage_criterion(cfg):
    """The coarser stage criterion for the linear phase, or ``None``.

    Strategy 1 asks for a *coarse* linear PGD before the corrections start: the
    linear modes only have to lay out the space that the polynomial terms will
    then curve, so spending a fine tolerance on them is spending it in the wrong
    place. Returns ``None`` when ``linear_stage_tol`` is unset, which is the
    signal to reuse the single criterion for every stage.

    Only the tolerance changes; the window, the iteration bounds and the floor
    are shared, so a run cannot end up comparing two differently-budgeted phases.

    Args:
        cfg (RunConfig): supplies ``linear_stage_tol`` and the shared bounds.

    Returns:
        RelativeChange | None
    """
    if cfg.linear_stage_tol is None:
        return None
    return _stage_criterion(cfg, cfg.linear_stage_tol)


def _stage_criterion(cfg, tol):
    return RelativeChange(
        tol=tol,
        window=cfg.window,
        max_iter=cfg.max_iter,
        min_iter=cfg.min_iter,
        floor=cfg.stage_floor,
    )


def build_optimizer_factory(cfg, decomposition=None):
    """Adam factory -- one optimizer per stage, per the base trainer.

    The monoms get ``cfg.lr``; the coefficient rows get ``cfg.coefficient_lr``,
    in their own param group. **This split is not a refinement, it is what makes
    the polynomial schedules run at all.**

    Why, measured. ``renormalise`` leaves the mode's whole amplitude on the space
    axis, so after a converged CP stage the monom norms are
    ``(2.9e5, 1, 1, 1, 1)`` and the terms' natural sizes are

        prod_j ||w_j||^lambda_j  =  2.9e5   (leading)
                                    8.4e10  (lambda = 2)
                                    2.5e16  (lambda = 3)

    For a correction to be a *correction* rather than a takeover, ``C_lambda``
    must therefore sit around ``A^(1 - p)`` -- about ``3e-6`` for ``p = 2`` and
    ``1e-11`` for ``p = 3``, where ``A`` is the mode amplitude. Adam's step is
    ``lr * m / sqrt(v) ~ lr`` regardless of the gradient's magnitude, so a shared
    ``lr = 0.1`` moves ``C`` by ~0.1 on its first step: five to ten orders of
    magnitude past the target. Measured on a tiny mesh, 80-iteration stages,
    2 modes -- final energy (lower is better; the CP baseline is ``-1.875e11``):

        lr_C      staged        joint
        1e-1     +2.2e18      -1.54e11   <- staged diverges, joint beats nothing
        1e-3     -1.91e11     -8.24e11
        1e-6     -1.89e11     -1.93e11
        1e-9     -1.88e11     -1.88e11   <- C too small to do anything

    Note this scale problem is **not** a gauge artefact and cannot be fixed by
    ``renormalise``: for a uniform exponent row, ``prod_j (s_j w_j)^p =
    (prod_j s_j)^p prod_j w_j^p = prod_j w_j^p`` because the gauge requires
    ``prod_j s_j = 1``. Those terms are exactly gauge-invariant. The scale is set
    by the mode's physical amplitude, and the proper fix is to reparameterise
    ``C`` against it inside ``PolynomialNLPGD`` -- not done, see the CHANGELOG.

    The leading coefficients ``c_i``, when the decomposition has them, join the
    **same** group as ``C`` rather than getting a third ``lr``. That is not
    laziness: they are only ever released alongside the all-``d``-axes gauge fix,
    under which every monom has unit norm and therefore every term -- leading
    included -- has natural size 1. The whole point of releasing ``c_i`` is that
    the two families stop living on different scales, so giving them different
    rates would undo it. If a measurement ever shows they need separate rates,
    that is evidence against the gauge argument and belongs in the CHANGELOG.

    Args:
        cfg (RunConfig): supplies ``lr`` and ``coefficient_lr``.
        decomposition (PolynomialNLPGD, optional): needed to tell a coefficient
            row from a monom. When omitted, every parameter gets ``cfg.lr`` --
            the plain single-group factory of the CP example.

    Returns:
        Callable: ``params -> torch.optim.Adam``.
    """
    if decomposition is None:
        return lambda params: torch.optim.Adam(params, lr=cfg.lr)

    coefficient_ids = {id(row) for row in decomposition.coefficients}
    if decomposition.has_leading_coefficients:
        coefficient_ids |= {id(c) for c in decomposition.leading_coefficients}

    def factory(params):
        monoms = [p for p in params if id(p) not in coefficient_ids]
        coefficients = [p for p in params if id(p) in coefficient_ids]
        groups = []
        if monoms:
            groups.append({"params": monoms, "lr": cfg.lr})
        if coefficients:
            groups.append({"params": coefficients, "lr": cfg.coefficient_lr})
        return torch.optim.Adam(groups)

    return factory


@dataclass
class Problem:
    """Everything ``build_problem`` assembles, kept together for the caller.

    Attributes:
        model (NeuROMModel): the trainable model (``model()`` fills the layout).
        pgd (PolynomialNLPGD): the separated representation. Kept under the name
            ``pgd`` rather than something format-specific so the sweep, the
            plotting helpers and the tests read identically to the CP example's.
        field_layout (FieldLayout): holds the monom fields and the load.
        domain (IntegrationDomain): interpolates every active field.
        axes (dict[str, Axis]): the five axes, keyed by name.
        history (TrainingHistory | None): Filled by ``main`` when it trains;
            None when the problem is only assembled.
    """

    model: NeuROMModel
    pgd: PolynomialNLPGD
    field_layout: FieldLayout
    domain: IntegrationDomain
    axes: dict[str, Axis]
    history: object = None


def build_problem(
    loss_fn, *, n_modes_max=N_MODES_MAX, n_modes_ini=1, n_nodes=None, quad=None,
    seed_amplitude=0.05, exponents=None, leading_coefficients=False,
    orthogonal_corrections=False,
):
    """Assemble the five axes, the polynomial NL-PGD, the load and the model.

    The energy is *injected*: this function never references ``energy`` directly,
    so the same wiring drives a different functional unchanged.

    Args:
        loss_fn (Callable): ``loss_fn(field_layout, decomposition) -> Tensor``.
        n_modes_max (int): Mode budget of the decomposition.
        n_modes_ini (int): Number of initially active (trainable) modes.
        exponents (torch.Tensor, optional): the exponent set ``I``, ``(T, 5)``.
            Defaults to :func:`build_exponents` on a default ``RunConfig``, i.e.
            ``uniform_exponents(5, 3)``.
        n_nodes (dict[str, int], optional): Per-axis node counts overriding
            ``DEFAULT_N_NODES``; used by the tests to build a tiny problem.
        quad (QuadratureRule, optional): Shared quadrature rule for every axis;
            defaults to ``MidPoint1D()`` (one point per element). Injectable so
            tests can exercise ``N_q > 1`` rules (e.g. ``TwoPoints1D``), which
            catch broadcasting bugs that a single quadrature point hides.
        seed_amplitude (float): Initial amplitude of every mode after the first,
            realised as an amplitude/shape split of the CP seed (see below).
        leading_coefficients (bool): Give each mode a trainable weight ``c_i`` on
            its own linear term. Built frozen at 1, so this alone changes
            nothing; a trainer has to release it. See ``RunConfig``.

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
    # Mode 0 carries the bulk of the solution, so it keeps the plain 0.5 seed on
    # every factor. Every *later* mode is seeded as an amplitude/shape split: its
    # parametric factors start at unit shape (1.0) and the whole product's small
    # initial amplitude is carried by the space factor (`seed_amplitude`). This
    # lets a new mode enter near zero -- no loss spike on enrichment -- while
    # keeping every factor strictly non-zero, so no gradient is locked at the
    # all-zero stationary point (cf. CPPGD.add_mode). Folding the amplitude into
    # a single factor exploits the CP scale degeneracy (see plot_modes); the
    # space factor is chosen because that is where the tutor's "seed the space
    # mode small" intuition lives, minus the collapse that seeding it at exactly
    # 0 causes.
    init_values = {name: 0.5 for name in AXIS_ORDER}
    init_values_rest = {name: 0.5 for name in AXIS_ORDER}
    init_values_rest["space"] = seed_amplitude
    axes = [
        make_axis(
            name,
            *AXIS_BOUNDS[name],
            counts[name],
            constraints[name],
            sf,
            quad,
            mapping,
            init_values[name],
            init_values_rest.get(name),
        )
        for name in AXIS_ORDER
    ]

    # Every coefficient row starts frozen at zero, so a freshly built problem is
    # numerically indistinguishable from the CP one in the sibling directory --
    # the polynomial terms only exist once a trainer releases a row. See
    # PolynomialGreedyTrainer.
    if exponents is None:
        exponents = build_exponents(RunConfig(name="default"))
    pgd = PolynomialNLPGD(
        axes=axes,
        n_modes_max=n_modes_max,
        exponents=exponents,
        name="pgd",
        n_modes_ini=n_modes_ini,
        leading_coefficients=leading_coefficients,
        orthogonal_corrections=orthogonal_corrections,
    )

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
    """Total potential energy of the tanh-graded bar, in polynomial separated form.

    The CP version of this function loops over pairs of **modes**. With a
    polynomial decomposition a mode is a *sum* of ``1 + |I|`` products, so the
    unit of summation becomes the **term** and the double loop runs over pairs of
    terms -- read from
    :meth:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD.polynomial_directory`,
    which supplies each term's mode, its per-axis exponents and its coefficient
    (``None`` for the leading term, whose weight is a fixed 1). Everything else
    is unchanged: each term is still a plain product over axes, so with

        E(x, E1, E2, alpha, n) = (E2 + E1)/2 + (E2 - E1)/2 tanh(n (x - alpha)),

    every factor of the elastic term separates into a product of 1-D integrals
    *except* tanh(n (x - alpha)), which couples x, alpha and n. That coupled block
    is integrated by an exact 3-D tensor-product quadrature over those three
    axes' quadrature points (one einsum per term pair); the (E2 +/- E1)/2
    prefactors stay 1-D moments of the E1 and E2 factors. The moments merely gain
    exponents: ``int lambda_i^a lambda_j^b dE1`` where CP had ``int lambda_i
    lambda_j dE1``.

    The one genuinely new subtlety is the **space gradient**. For a term
    ``C prod_j w_j^lambda_j`` the chain rule gives

        d/dx [ C X^p (...) ] = C p X^(p-1) X' (...),

    so the powered gradient must be built from *both* the interpolated value
    ``X = r.u`` and ``jacobian_field`` -- differentiating the CP way, from the
    jacobian alone, silently drops the ``p X^(p-1)`` factor. Here that factor is a
    scalar ``(N_e, N_q, 1)`` prefactor multiplied onto the density *after*
    ``inner`` has contracted the jacobian's trailing axis, which keeps the
    contraction exactly as the CP example does it and avoids reintroducing the
    reshape trap noted below.

    Cost
        The double loop is quadratic in ``n_modes * (1 + |I|)``. At the default
        ``I`` (two correction terms) five modes give 15 terms, so 225 pairs
        against CP-PGD's 25 at the same rank. This is why ``N_MODES_MAX`` is 5.

    Sign convention follows the CP example: the returned value is
    ``elastic + load``, the load field carrying its own sign.

    Args:
        field_layout (FieldLayout): filled by ``model()`` (train mode).
        decomposition (PolynomialNLPGD): supplies the active monom names via
            ``directory()`` and the term structure via ``polynomial_directory()``.
        load_name (str): name of the load field in the layout.

    Returns:
        torch.Tensor: 0-dim energy.
    """
    directory = decomposition.directory()
    # `skip_inert`: drop correction terms frozen at exactly zero. The double loop
    # below is quadratic in `len(terms)`, and it has no zero-test of its own -- a
    # zero coefficient builds its whole product and then multiplies it by 0. So a
    # linear-phase stage (every `C` frozen at 0) was paying the full
    # `(n_modes * (1 + |I|))^2`: measured at |I| = 5, three linear modes cost
    # 5.6/21.2/49.3 s against 1.8/5.4/11.5 s at |I| = 2, the 4x the term ratio
    # predicts. The field is identical either way; only what is assembled moves.
    terms = decomposition.polynomial_directory(skip_inert=True)

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
    # first moments int E1 lambda_i^a lambda_j^b dE1 and its E2 twin).
    lam, E1_val, J1 = [r.u for r in e1], [r.x for r in e1], [r.measure for r in e1]
    mu, E2_val, J2 = [r.u for r in e2], [r.x for r in e2], [r.measure for r in e2]
    A, Ja = [r.u for r in alp], [r.measure for r in alp]
    N, Jn = [r.u for r in slp], [r.measure for r in slp]

    # Per-term quantities, hoisted out of the quadratic loop: every power is
    # raised ONCE here rather than O(n_terms) times inside it. `gpref` is the
    # scalar C p X^(p-1) of the chain rule above; the jacobian itself stays
    # separate so that inner() still does the contraction.
    mode_of = []
    gpref, xval, lval, mval, aval, nval = [], [], [], [], [], []
    for m, lamb, coefficient in terms:
        weight = 1.0 if coefficient is None else coefficient
        p = lamb[0]
        mode_of.append(m)
        gpref.append(weight * p * X[m] ** (p - 1))
        xval.append(weight * X[m] ** p)
        lval.append(lam[m] ** lamb[1])
        mval.append(mu[m] ** lamb[2])
        aval.append(A[m] ** lamb[3])
        nval.append(N[m] ** lamb[4])

    # NB: as in the CP example, the cross terms assume the two terms' modes share
    # a mesh per axis (the measure and coordinates of the first are used for
    # both). Independent per-mode meshes would need a common intersection mesh
    # with a recomputed measure. tanh_grid below is the more fragile consumer of
    # this assumption, since it hard-codes mode 0's quadrature points for every
    # pair.
    #
    # The one non-separable block: tanh(n (x - alpha)) on the tensor product of
    # the space, alpha and n quadrature points, shape (Qx, Qalpha, Qn). It does
    # not depend on the term pair, so it is built once and reused below.
    # Rebuilt every call (rather than cached at setup) so it stays correct if the
    # meshes ever become trainable (r-adaptivity).
    xq = spc[0].x.reshape(-1)
    aq = alp[0].x.reshape(-1)
    nq = slp[0].x.reshape(-1)
    tanh_grid = torch.tanh(nq[None, None, :] * (xq[:, None, None] - aq[None, :, None]))

    elastic = 0.0
    for t, i in enumerate(mode_of):
        for s, j in enumerate(mode_of):
            # Densities at the quadrature points, reused both integrated (the
            # separable part) and raw (contracted against tanh_grid). Both terms'
            # coefficients and chain-rule factors ride in gpref.
            kx_density = gpref[t] * gpref[s] * inner(gX[i], gX[j]) * Jx[i]
            a_density = aval[t] * aval[s] * Ja[i]
            n_density = nval[t] * nval[s] * Jn[i]

            Kx = integrate(kx_density)
            P0 = integrate(a_density)
            Q0 = integrate(n_density)
            L0 = integrate(lval[t] * lval[s] * J1[i])
            L1 = integrate(E1_val[i] * lval[t] * lval[s] * J1[i])
            M0 = integrate(mval[t] * mval[s] * J2[i])
            M1 = integrate(E2_val[i] * mval[t] * mval[s] * J2[i])

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

    # Constant load: linear in u, so one term at a time -- separable across every
    # axis, one 1-D integral per factor. The term's coefficient is already in
    # xval.
    load_f = field_layout[load_name].u
    load = 0.0
    for t, i in enumerate(mode_of):
        Fx = integrate(inner(load_f, xval[t]) * Jx[i])
        load = load + (
            Fx
            * integrate(lval[t] * J1[i])
            * integrate(mval[t] * J2[i])
            * integrate(aval[t] * Ja[i])
            * integrate(nval[t] * Jn[i])
        )

    return elastic + load


## Training strategy


#: The three ways a mode's monoms and its coefficient row can be fitted, all of
#: them purely greedy (a finished mode is frozen for good, coefficients
#: included). ``joint`` and ``staged`` are the pair the example exists to
#: compare; ``refine`` is the intermediate the ``PolynomialNLPGD`` docstring
#: describes, kept because it is the obvious "best of both" guess and therefore
#: the obvious thing to be wrong about.
#:
#: Each entry maps to (stages per mode, what is trainable in each stage).
SCHEDULES = {
    # One stage per mode. The monoms and C descend the same energy together from
    # the mode's first iteration.
    "joint": ("joint",),
    # Two stages. CP first (C frozen at 0), then the monoms are frozen at their
    # converged values and C alone is fitted on top.
    "staged": ("cp", "corr"),
    # Two stages. CP first, then monoms AND C together.
    "refine": ("cp", "joint"),
    # Two stages. CP first, then the SPACE monom is frozen as a fixed support
    # and only the parametric monoms and C are fitted on top. The non-linearity
    # curves the parameter directions of a space the linear mode has laid out,
    # rather than acting on the space factor itself.
    "support": ("cp", "support"),
}

#: A mode in the linear phase (``RunConfig.n_linear_modes``) costs one pure-CP
#: stage whatever the schedule -- that is what "linear phase" means.
LINEAR_SCHEDULE = ("cp",)


class PolynomialGreedyTrainer(GreedyTrainer):
    """Purely greedy enrichment that also fits the polynomial coefficients.

    Why this exists at all
        ``PolynomialNLPGD`` zero-initialises every coefficient row and leaves it
        with ``requires_grad=False``; ``freeze_all()`` re-freezes them. Neither
        :class:`~neurom.training.GreedyTrainer` nor
        :class:`~neurom.training.SimultaneousTrainer` ever calls
        ``unfreeze_mode_coefficients``, so driving this decomposition with either
        of them leaves ``C = 0`` for the whole run -- a result **bit-identical to
        CP-PGD**. Useful as a control, useless as a test of the format. Somebody
        has to release the rows; this is that somebody.

    Greedy, in every schedule
        A mode that is finished is frozen for good -- ``freeze_all()`` covers the
        monoms *and* the coefficients, so a released row is re-frozen the moment
        the next mode is added and is never revisited. Mode ``m`` therefore
        always descends the residual its predecessors left, exactly as in the CP
        example. The schedules differ only in how mode ``m``'s own parameters are
        fitted before that freeze.

    The three schedules (:data:`SCHEDULES`)
        ``joint`` -- one stage per mode; the monoms and ``C`` train together from
        the mode's first iteration. ``C`` starts at 0, but ``dE/dC`` does **not**
        vanish there (the energy is quadratic in ``u`` and ``u`` is affine in
        ``C``), so the row takes off on its own; it is only ``dE/dw`` that is
        insensitive to a zero ``C``, which means the monoms spend the first
        iterations descending the plain CP gradient and are progressively pulled
        by the correction as it grows. Fully non-convex, and the gauge
        degeneracy is live throughout -- this is the schedule ``renormalise``
        exists for.

        ``staged`` -- two stages: mode ``m``'s monoms alone (with ``C`` frozen at
        0, so the stage sees exactly the CP energy), then the monoms frozen at
        their converged values and ``C`` alone fitted on top. The second stage is
        a **quadratic** minimisation in ``|I|`` unknowns -- with the monoms
        fixed, ``u`` is affine in ``C`` and the energy is quadratic in ``u``, so
        it has a unique minimum that could in principle be solved directly rather
        than by Adam. It is also the honest answer to "what does the polynomial
        term add on top of a *converged* CP mode?", since the CP half is
        identical to what the CP baseline does.

        ``refine`` -- two stages: CP first, then monoms and ``C`` together. The
        protocol :class:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD`
        documents.

        ``support`` -- two stages: CP first, then the **space monom is frozen**
        and only the parametric monoms and ``C`` are fitted on top. The linear
        mode fixes a support; the correction curves the parameter directions over
        it, rather than reshaping the space factor. Its natural companion is
        ``RunConfig.pin_space_exponent``, which keeps the space exponent at 1 so
        the corrections are non-linear in the parameters alone -- but the two are
        independent knobs, so the sweep can tell which of them does the work.

        Note that ``staged``'s and ``support``'s frozen monoms are still
        *rescaled* by the gauge fix, which runs before the optimizer is built and
        is field-preserving. What "frozen" promises is that the optimizer does
        not touch them.

    The linear phase (``n_linear_modes``)
        Orthogonal to the schedule. The first ``l`` modes are trained as pure CP,
        one stage each, and only the modes after them run the schedule above.
        This is the "curve a space a linear PGD has already laid out" reading:
        the non-linearity never acts from a mode's seed, only on top of an
        existing linear decomposition. With ``linear_stage_criterion`` that phase
        can also run coarser.

        Combined with ``support`` it is the fusion the plan settles on -- a few
        linear modes, then frozen-support corrections -- and it needs no
        additional code, which is the point of keeping the two independent.

    The leading coefficient (``leading_coefficient``)
        When on, a non-linear mode's own linear term carries a trainable weight
        ``c_i``, released alongside its ``C`` row, so a late mode may have little
        or no linear part at all. Requires a decomposition built with
        ``leading_coefficients=True``. Modes inside the linear phase never get
        one. Note this is a *reparameterisation*, not extra expressivity -- see
        :class:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD`.

    Enrichment bookkeeping
        The enrichment criterion is fed the **end-of-mode** records only, via
        :meth:`mode_final_stages`. Handed the raw list under a two-stage
        schedule, a ``RelativeGain`` would compare a correction stage against the
        CP stage of the *same* mode and read the (small) correction gain as
        convergence after one mode. Consequently ``MaxStages(n)`` handed to this
        trainer means **n modes**, not n stages, under every schedule -- which is
        also what makes the schedules comparable at equal rank.

        This is a *lookup*, not the arithmetic slice it replaces: with a linear
        phase, modes no longer all cost the same number of stages, so
        ``stages[spm - 1 :: spm]`` would silently pick the wrong records.

    Diagnostics
        Inherited unchanged, and therefore CP-only: ``amplitude`` and
        ``max_correlation`` are computed from the monom values and know nothing
        about ``C``. Two modes reported as uncorrelated could still carry similar
        corrections. ``_report`` prints the coefficient rows separately.

    Args:
        model (NeuROMModel): model whose ``decomposition`` is a
            :class:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD`.
        schedule (str): a key of :data:`SCHEDULES`. Defaults to the class
            attribute, which the concrete subclasses below pin.
        n_linear_modes (int): how many leading modes are pure CP, one stage each,
            before the schedule starts. Defaults to 0.
        leading_coefficient (bool): release ``c_i`` on the non-linear modes.
            Defaults to False; raises if the decomposition has none.
        linear_stage_criterion (StageCriterion, optional): stage criterion for
            the linear phase only. ``None`` reuses ``stage_criterion``.
        **kwargs: as :class:`~neurom.training.base.PGDTrainer`.
    """

    #: Default schedule; overridden by the subclasses so that ``STRATEGIES`` can
    #: map a CLI name to a class whose ``__name__`` names a checkpoint file.
    schedule = "joint"

    def __init__(
        self,
        model,
        schedule=None,
        n_linear_modes=0,
        leading_coefficient=False,
        linear_stage_criterion=None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        if schedule is not None:
            self.schedule = schedule
        if self.schedule not in SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; "
                f"pick one of {sorted(SCHEDULES)}"
            )
        self.n_linear_modes = int(n_linear_modes)
        self.leading_coefficient = bool(leading_coefficient)
        self.linear_stage_criterion = linear_stage_criterion
        if self.leading_coefficient and not self.decomposition.has_leading_coefficients:
            raise ValueError(
                "leading_coefficient=True needs a decomposition built with "
                "PolynomialNLPGD(..., leading_coefficients=True); this one has a "
                "fixed leading weight of 1."
            )
        # Stage index -> (mode, kind). Built one whole mode at a time by
        # `_ensure_plan`, so the plan always ends on a mode boundary.
        self._plan = []

    @property
    def stages_per_mode(self):
        """How many stages one *non-linear* mode costs (1 or 2).

        The schedule's own length. With a linear phase this is no longer the
        stage-to-mode ratio of the run -- linear modes cost one stage each
        whatever the schedule -- so index arithmetic must go through
        :meth:`mode_of_stage`, not through this.
        """
        return len(SCHEDULES[self.schedule])

    # -- the stage plan --------------------------------------------------------

    def _schedule_for_mode(self, mode):
        """The stage kinds mode ``mode`` costs: linear phase, then the schedule."""
        if mode < self.n_linear_modes:
            return LINEAR_SCHEDULE
        return SCHEDULES[self.schedule]

    def _ensure_plan(self, stage_index):
        """Extend the plan, a whole mode at a time, to cover ``stage_index``.

        Modes do **not** all cost the same number of stages once a linear phase
        is in play, so the stage -> mode map cannot be arithmetic. Appending
        entire modes keeps the invariant every consumer relies on: the plan never
        ends part-way through a mode, so its last entry is always mode-final.
        """
        while len(self._plan) <= stage_index:
            mode = self._plan[-1][0] + 1 if self._plan else 0
            for kind in self._schedule_for_mode(mode):
                self._plan.append((mode, kind))

    def mode_of_stage(self, stage_index):
        """Which mode stage ``stage_index`` belongs to."""
        self._ensure_plan(stage_index)
        return self._plan[stage_index][0]

    def stage_kind(self, stage_index):
        """What stage ``stage_index`` trains.

        * ``cp`` -- the mode's monoms, coefficients frozen at 0.
        * ``corr`` -- the coefficient row alone, every monom frozen.
        * ``joint`` -- monoms and coefficients together.
        * ``support`` -- the *parametric* monoms and the coefficients, with the
          space monom frozen as a fixed support.
        """
        self._ensure_plan(stage_index)
        return self._plan[stage_index][1]

    def adds_a_mode(self, stage_index):
        """Whether this stage is a mode's first (and so calls ``add_mode``)."""
        self._ensure_plan(stage_index)
        return stage_index == 0 or self._plan[stage_index - 1][0] != self._plan[stage_index][0]

    def mode_final_stages(self, before):
        """Indices of the stages that *finish* a mode, among those before ``before``.

        What the enrichment criterion must be fed. Handed the raw stage list, a
        ``RelativeGain`` under a two-stage schedule would compare a mode's
        correction stage against its own CP stage and read the (small) correction
        gain as convergence after one mode. This replaces the old
        ``[spm - 1 :: spm]`` slice, which assumed every mode costs the same.
        """
        self._ensure_plan(max(before - 1, 0))
        return [
            i
            for i in range(min(before, len(self._plan)))
            if i + 1 >= len(self._plan) or self._plan[i + 1][0] != self._plan[i][0]
        ]

    # -- the trainer hooks -----------------------------------------------------

    def releases_the_leading_coefficient(self, mode):
        """Whether mode ``mode`` gets its ``c_i`` released.

        Only outside the linear phase: the linear modes are meant to be plain CP,
        and releasing ``c`` there would let one of them absorb an amplitude that
        the gauge fix already handles.
        """
        return self.leading_coefficient and mode >= self.n_linear_modes

    def prepare_stage(self, stage_index):
        """Set the freeze state this stage's kind calls for, then build the optimizer.

        Args:
            stage_index (int): index of the stage about to run.
        """
        mode = self.mode_of_stage(stage_index)
        kind = self.stage_kind(stage_index)

        if self.adds_a_mode(stage_index):
            # Purely greedy: freeze everything -- monoms AND coefficients, which
            # is what this decomposition's freeze_all covers -- then activate one
            # mode, which unfreezes its monoms only. Stage 0 adds nothing; the
            # decomposition arrives with mode 0 active and its monoms trainable.
            if stage_index > 0:
                self.decomposition.freeze_all()
                self.decomposition.add_mode()
        if kind == "corr":
            self.decomposition.freeze_mode(mode)
        if kind == "support":
            # The space factor is the support: fixed, with the correction free to
            # curve the parameter directions on top of it. Unfreeze first, so a
            # `support` stage reached from a frozen state still releases the
            # parametric monoms.
            self.decomposition.unfreeze_mode(mode)
            self.decomposition.freeze_monom(mode, AXIS_ORDER.index("space"))
        if kind in ("corr", "joint", "support"):
            self.decomposition.unfreeze_mode_coefficients(mode)
            if self.releases_the_leading_coefficient(mode):
                self.decomposition.unfreeze_mode_leading_coefficient(mode)

        # Before make_optimizer, never after: the gauge fix rescales parameters,
        # so Adam's moments would refer to a state that no longer exists.
        self.fix_gauge()
        self.make_optimizer()

    def stage_criterion_for(self, stage_index):
        """The stage criterion this stage runs under.

        The linear phase may run coarser (see
        :func:`build_linear_stage_criterion`); everything else uses the single
        injected criterion.
        """
        if (
            self.linear_stage_criterion is not None
            and self.mode_of_stage(stage_index) < self.n_linear_modes
        ):
            return self.linear_stage_criterion
        return self.stage_criterion

    def stage(self, stage_index):
        """Run one stage under whichever criterion its phase calls for.

        The base class reads ``self.stage_criterion`` directly, so the phase
        switch is done by swapping it around the call rather than by duplicating
        the loop. Restored in a ``finally`` so a diverged stage cannot leave the
        coarse criterion in place for the rest of the run.
        """
        criterion = self.stage_criterion_for(stage_index)
        if criterion is self.stage_criterion:
            return super().stage(stage_index)
        saved, self.stage_criterion = self.stage_criterion, criterion
        try:
            return super().stage(stage_index)
        finally:
            self.stage_criterion = saved

    def should_add_stage(self, stage_index):
        """Run every stage of each mode; enrich until capacity or a small gain.

        Only a mode's *first* stage is subject to capacity and to the enrichment
        criterion; the later stage of a two-stage schedule adds nothing, it
        finishes the mode its predecessor started.

        Args:
            stage_index (int): index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """
        if not self.adds_a_mode(stage_index):
            return True
        if stage_index == 0:
            return True
        if self.decomposition.n_modes_truncated >= self.decomposition.n_modes_max:
            self.history.stop_reason = "capacity"
            return False
        finals = self.mode_final_stages(stage_index)
        reason = self.enrichment_criterion.stop_reason(
            [self.history.stages[i] for i in finals]
        )
        if reason:
            self.history.stop_reason = reason
            return False
        return True

    def stage_label(self, stage_index):
        """The stage's kind, so the live bar says which half of a schedule runs.

        Same string ``on_stage_end`` records, from the same ``stage_kind``: with
        ``refine`` and ``staged`` spending two stages per mode, an index alone
        does not say whether the bar is on the CP stage or the correction.

        Args:
            stage_index (int): index of the stage about to run.

        Returns:
            str: ``"cp"``, ``"corr"`` or ``"joint"``.
        """
        return self.stage_kind(stage_index)

    def on_stage_end(self, record):
        """Inherited CP diagnostics, plus what this stage trained and its L1 norm.

        ``kind`` is what lets ``_report``'s table label a row without
        re-deriving it from an index parity it does not own. ``coefficient_norm``
        is the one diagnostic that looks at ``C`` at all: a stage that ends with
        it still at zero did not fit a correction, whatever the energy did.

        Args:
            record (StageRecord): the stage that just finished.
        """
        super().on_stage_end(record)
        record.diagnostics["kind"] = self.stage_kind(record.stage)
        mode = self.mode_of_stage(record.stage)
        record.diagnostics["mode"] = mode
        record.diagnostics["coefficient_norm"] = float(
            self.decomposition.coefficients[mode].detach().abs().sum()
        )
        if self.decomposition.has_leading_coefficients:
            record.diagnostics["leading_coefficient"] = float(
                self.decomposition.leading_coefficients[mode].detach()
            )
            # Every active mode's `c`, not just this stage's. A greedy stage only
            # trains one mode, but `fix_gauge` runs over all of them, so an
            # earlier mode's `c` keeps moving long after its own stage ended --
            # that drift is invisible in the scalar above, which reports the
            # active mode alone. This is the column to read to see whether a `c`
            # settled or is still climbing (see `sweep.show_leading_coefficients`).
            record.diagnostics["leading_coefficients"] = [
                float(self.decomposition.leading_coefficients[m].detach())
                for m in range(int(self.decomposition.n_modes_truncated))
            ]


class JointNLGreedyTrainer(PolynomialGreedyTrainer):
    """Greedy, one stage per mode, monoms and coefficients trained together."""

    schedule = "joint"


class StagedNLGreedyTrainer(PolynomialGreedyTrainer):
    """Greedy; CP mode first, then its coefficient row alone with monoms frozen."""

    schedule = "staged"


class RefineNLGreedyTrainer(PolynomialGreedyTrainer):
    """Greedy; CP mode first, then monoms and coefficients together."""

    schedule = "refine"


class SupportNLGreedyTrainer(PolynomialGreedyTrainer):
    """Greedy; CP mode first, then its space monom is frozen as a fixed support.

    The second stage fits the parametric monoms and the coefficient row over a
    space factor that no longer moves -- the non-linearity curves the parameter
    directions of a support the linear stage laid out.
    """

    schedule = "support"


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

    path = REFERENCE_DIR / "reference_fem_solution.py"
    spec = importlib.util.spec_from_file_location("reference_fem_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reference()


def plot_convergence(history, save_path=None):
    """Plot the energy against the training iteration, with the stage boundaries.

    Two rule styles, because a polynomial run has two kinds of boundary: a solid
    grey rule where a **mode** is added (the start of a CP stage) and a dotted one
    where that mode's **coefficient row** is released. Reading a single style
    would make a two-stage run look like a run of twice as many modes. A run
    under a CP baseline has no correction stages, so every rule is solid.

    Args:
        history (TrainingHistory): filled by the trainer's ``enrich``.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "nlpgd5_convergence.png"``.
    """
    import matplotlib.pyplot as plt

    save_path = save_path or PLOT_DIR / "nlpgd5_convergence.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history.losses)), history.losses, "b-")

    boundary = 0
    for record, following in zip(history.stages, history.stages[1:]):
        boundary += record.n_iter
        correction = following.diagnostics.get("kind") == "corr"
        ax.axvline(
            boundary,
            color="grey",
            ls=":" if correction else "-",
            lw=1,
            alpha=0.6 if correction else 0.9,
        )

    ax.set_title("" \
    "Training convergence")
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
        pgd (PolynomialNLPGD): the trained decomposition.
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
    n_points, n_x = u_ref.shape

    # One query matrix for *all* reference points, not one evaluate() per point:
    # ``pgd.evaluate`` is already vectorised over its rows, so the K x P table is
    # a single (K*P, 5) call. Looping in Python instead cost a full mesh search
    # per point -- fine at a dozen points, but a stall at the 627-point grid.
    columns = {"space": x_ref.unsqueeze(0).expand(n_points, n_x)}  # (K, P)
    columns.update(
        {
            name: reference["params"][:, j].unsqueeze(1).expand(n_points, n_x)
            for j, name in enumerate(reference["param_names"])
        }
    )
    query = torch.stack([columns[name] for name in AXIS_ORDER], dim=2)  # (K, P, 5)
    u_pgd = pgd.evaluate(query.reshape(-1, len(AXIS_ORDER))).reshape(n_points, n_x)

    per_point = {
        label: (
            torch.linalg.norm(u_pgd[k] - u_ref[k]) / torch.linalg.norm(u_ref[k])
        ).item()
        for k, label in enumerate(reference["labels"])
    }
    overall = (torch.linalg.norm(u_pgd - u_ref) / torch.linalg.norm(u_ref)).item()
    return {"per_point": per_point, "overall": overall, "u_pgd": u_pgd}


def plot_solution(pgd, reference=None, labels=None, save_path=None):
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
        pgd (PolynomialNLPGD): the trained decomposition.
        reference (dict, optional): a loaded reference bundle; by default the
            one on disk next to this script.
        labels (list[str], optional): which reference points to draw; defaults
            to the bundle's highlighted ones.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "nlpgd5_vs_reference.png"``.

    Returns:
        dict: the ``relative_errors`` result, computed over *all* the reference
        points, not only the plotted ones.
    """
    import matplotlib.pyplot as plt

    save_path = save_path or PLOT_DIR / "nlpgd5_vs_reference.png"
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


def plot_modes(pgd, save_path=None):
    """Plot every mode's **monom** on every axis, normalised by its max modulus.

    A mode is defined up to a per-axis scale (multiply one factor by ``c``,
    divide another by ``c`` and the product is unchanged -- here with the extra
    ``prod_j s_j = 1`` constraint, and the coefficients absorbing the rest), so
    only the *shape* of a factor is meaningful -- hence the normalisation. Two
    modes whose curves coincide on every axis are the degenerate case the
    ``max_correlation`` diagnostic reports.

    What this figure does **not** show: the coefficients. A mode's actual
    contribution is ``prod_j w_j + sum_lambda C_lambda prod_j w_j^lambda_j``, and
    two modes with identical monoms could still differ through ``C``. The
    coefficient rows are printed by ``_report`` instead; read the two together.

    Args:
        pgd (PolynomialNLPGD): the trained decomposition.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "nlpgd5_modes.png"``.
    """
    import matplotlib.pyplot as plt

    from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

    save_path = save_path or PLOT_DIR / "nlpgd5_modes.png"

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


def main(
    verbose=True,
    train=True,
    plot=True,
    trainer_cls=JointNLGreedyTrainer,
    stage_min_iter=None,
    checkpoint=None,
    retrain=False,
    config=None,
):
    """Assemble the 5-parametric problem, evaluate the energy once, and train it.

    Always builds the problem and evaluates the energy once at the initial
    (untrained) state -- this proves the whole chain (five axes -> polynomial
    NL-PGD -> shared IntegrationDomain -> separated energy) assembles and
    produces a finite value. Unless ``train=False``, it then trains the
    decomposition with ``trainer_cls`` and returns the ``Problem`` with
    ``history`` filled.

    Args:
        verbose (bool): print a short summary of the assembled problem and,
            if training, a per-stage diagnostics table.
        train (bool): if True, train the decomposition after the initial energy
            evaluation; if False, return the untrained problem (``history``
            stays ``None``).
        trainer_cls (type): the training strategy, constructed as
            ``trainer_cls(model, stage_criterion=..., enrichment_criterion=...)``
            -- one of :data:`STRATEGIES`' values. :class:`JointNLGreedyTrainer`
            (the default), :class:`StagedNLGreedyTrainer` and
            :class:`RefineNLGreedyTrainer` fit the coefficients on the three
            schedules of :data:`SCHEDULES`; :class:`~neurom.training.GreedyTrainer`
            and :class:`~neurom.training.SimultaneousTrainer` never release a
            coefficient row, so they are the CP baseline. All five are scored
            against the same saved FEM reference, which is the point of making
            this a knob.
        stage_min_iter (int, optional): iterations a stage must run before the
            plateau test applies. Defaults to ``STAGE_MIN_ITER[trainer_cls]``,
            which is not the same for every strategy -- see the comment at
            the trainer below.
        plot (bool): if True (and training ran), draw the convergence curve, the
            PGD-vs-reference comparison and the per-axis mode factors. Requires
            matplotlib, which is imported lazily inside the plotting helpers.
        checkpoint (str or Path, optional): where the trained model lives.
            Defaults to ``checkpoint_path(trainer_cls)``. If the file exists and
            ``retrain`` is False, it is **loaded instead of training** -- which
            is the point: iterating on a figure costs seconds, retraining costs
            minutes. Otherwise training runs and the result is written there.
            Pass ``checkpoint=False`` to disable the mechanism entirely.
        retrain (bool): train even if a checkpoint exists, and overwrite it.

    Returns:
        Problem: the assembled objects, for interactive use.
    """
    if stage_min_iter is None:
        stage_min_iter = STAGE_MIN_ITER.get(trainer_cls, 120)

    # A RunConfig is the single source of every training knob. When none is
    # given, synthesise the one that reproduces this example's historical
    # defaults, so the CLI and the sweep drive identical machinery.
    strategy_name = {v: k for k, v in STRATEGIES.items()}.get(trainer_cls, "joint")
    if config is None:
        config = RunConfig(
            name=strategy_name,
            strategy=strategy_name,
            stage_tol=1e-5,
            window=20,
            max_iter=max(600, 2 * stage_min_iter),
            min_iter=stage_min_iter,
            enrichment_tol=1e-3,
            n_nodes=DEFAULT_N_NODES,
        )

    problem = build_problem(
        lambda layout, pgd: energy(layout, pgd, load_name="load"),
        n_modes_max=config.n_modes_max,
        n_nodes=config.n_nodes,
        seed_amplitude=config.seed_amplitude,
        exponents=build_exponents(config),
        leading_coefficients=config.leading_coefficient,
        orthogonal_corrections=config.orthogonal_corrections,
    )

    field_layout = problem.model()
    value = problem.model.loss(field_layout)

    if verbose:
        n_nodes_actual = {
            name: problem.axes[name].topology.n_nodes for name in AXIS_ORDER
        }
        print("axes            :", [axis.name for axis in problem.pgd.axes])
        print("nodes per axis  :", n_nodes_actual)
        print("active modes    :", problem.pgd.n_modes_truncated)
        print(f"exponent set    : {config.exponent_set}(5, {config.max_power}) -> "
              f"{problem.pgd.n_terms} correction term(s) per mode")
        print("terms per mode  :", 1 + problem.pgd.n_terms)
        print(f"energy          : {value.item():.6e}")

    if not train:
        return problem

    if checkpoint is None:
        checkpoint = checkpoint_path(trainer_cls)

    # Train once, then plot as often as you like. The checkpoint carries the
    # monom values AND which modes are active (the `active` flags are buffers,
    # so they ride along in the state_dict), so a loaded model evaluates exactly
    # what the run produced. The history rides along too, which is what lets the
    # convergence plot and the diagnostics table be redrawn without retraining.
    if checkpoint and not retrain and Path(checkpoint).exists():
        history, metadata = load_checkpoint(checkpoint, problem.model)
        problem.history = history
        if verbose:
            print(f"loaded          : {checkpoint}")
            print(f"  trained with  : {metadata.get('strategy', '?')}")
            print(f"  active modes  : {problem.pgd.n_modes_truncated}")
            if metadata.get("n_nodes") != DEFAULT_N_NODES:
                # A shape mismatch would have raised; a *node-count* mismatch
                # cannot, since n_nodes is baked into the shapes -- but the
                # criteria or the strategy may still differ from what is asked
                # for now, and the plots would silently describe the old run.
                print(f"  WARNING: saved meshes {metadata.get('n_nodes')} differ")
        _report(problem, history, verbose=verbose, plot=plot)
        return problem

    # One mode per stage, under whichever freeze schedule trainer_cls imposes.
    #
    # min_iter is load-bearing twice over. Adam spends a long sticky early phase
    # on this energy where the loss barely moves, which a plateau detector reads
    # as convergence. And below ~80 iterations per stage the greedy step simply
    # rediscovers mode 0 -- max_correlation reads 1.0 and the "modes" are copies
    # of each other (see CHANGELOG). The printed max corr column is what tells
    # you whether that is happening.
    #
    # It is also the one setting the two strategies must NOT share. A greedy
    # stage spends its whole budget on one new mode; a simultaneous stage has to
    # re-fit every earlier mode as well, and at 120 iterations the new mode
    # never takes off (amplitude 3e1 against mode 0's 3e5, gain 1.7e7 against
    # greedy's 5.6e9) -- small enough that RelativeGain calls the run converged
    # after two stages. At 300 it does take off and overtakes greedy. See
    # STAGE_MIN_ITER and the CHANGELOG.

    stage_criterion, enrichment_criterion = build_criteria(config)
    kwargs = dict(
        optimizer_factory=build_optimizer_factory(config, problem.pgd),
        stage_criterion=stage_criterion,
        enrichment_criterion=enrichment_criterion,
        # A stage of this problem is minutes long; a silent run is
        # indistinguishable from a hung one. Stays off when not verbose, so
        # scripted runs and tests print nothing.
        progress=ProgressBar() if verbose else None,
        # Base-trainer knob, so it applies to the CP baselines too -- where it is
        # a no-op, `CPPGD.renormalise()` being empty.
        renormalise=config.renormalise,
    )
    # The two library trainers are the CP baseline and know none of these; a
    # config that asks for them under `greedy` is a config error, not something
    # to drop on the floor.
    if issubclass(trainer_cls, POLYNOMIAL_TRAINERS):
        kwargs.update(
            n_linear_modes=config.n_linear_modes,
            leading_coefficient=config.leading_coefficient,
            linear_stage_criterion=build_linear_stage_criterion(config),
        )
    elif config.n_linear_modes or config.leading_coefficient or config.linear_stage_tol:
        raise ValueError(
            f"{trainer_cls.__name__} is a CP baseline and ignores n_linear_modes, "
            "leading_coefficient and linear_stage_tol. Leave them at their "
            "defaults, or pick a polynomial strategy."
        )
    trainer = trainer_cls(problem.model, **kwargs)

    history = trainer.enrich()
    problem.history = history

    if checkpoint:
        save_checkpoint(
            checkpoint,
            problem.model,
            history,
            metadata={
                "strategy": trainer_cls.__name__,
                "n_nodes": config.n_nodes or DEFAULT_N_NODES,
                "stage_min_iter": config.min_iter,
                "n_modes": problem.pgd.n_modes_truncated,
                # Not redundant with the config below: the exponent set is baked
                # into the coefficient rows' SHAPE, so a checkpoint reloaded into
                # a differently-configured problem fails on a shape mismatch
                # rather than silently. Recorded so the message is readable.
                "exponents": problem.pgd.exponents.tolist(),
                "config": dataclasses.asdict(config),
            },
        )
        if verbose:
            print(f"saved           : {checkpoint}")

    _report(problem, history, verbose=verbose, plot=plot)
    return problem


def term_magnitudes(pgd, m):
    """Size of each of mode ``m``'s terms: ``|coeff| * prod_j ||w_mj||^lambda_j``.

    The quantity to read, rather than the raw ``c`` and ``C``, because it is
    **gauge-invariant**: rescaling ``w_mj -> s_j w_mj`` divides the coefficients
    by exactly the factor it multiplies the norms by, so these numbers do not
    move while ``c`` and ``C`` individually do. Raw coefficients are only
    comparable to each other after ``renormalise`` with a live leading
    coefficient, which puts every ``||w_mj||`` at 1 -- so under
    ``renormalise=False``, or under the ``d-1`` gauge fix, reading ``c`` against
    ``C`` compares numbers in different units.

    That makes it the pair of answers worth having:

    * **Did the non-linear correction activate?** Some correction term holds a
      non-negligible share. A ``|C|`` that is merely non-zero proves nothing --
      it can sit against monom norms that make its term irrelevant.
    * **Did the linear part collapse?** The leading share goes to ~0, i.e. the
      mode became essentially purely non-linear. That is a *reachable* state
      only with ``leading_coefficient=True``; with ``c`` pinned at 1 it is a
      limit point (monoms -> 0, ``C`` -> infinity), not an interior one.

    Args:
        pgd (PolynomialNLPGD): a decomposition, trained.
        m (int): mode index.

    Returns:
        tuple: ``(leading, corrections)`` -- a float, and one float per row of
        ``pgd.exponents``, in the same order.
    """
    norms = [float(n) for n in pgd.monom_norms(m)]
    weight = 1.0
    if pgd.has_leading_coefficients:
        weight = abs(float(pgd.leading_coefficients[m].detach()))
    leading = weight
    for value in norms:
        leading *= value

    corrections = []
    row = pgd.coefficients[m].detach()
    for t in range(pgd.n_terms):
        size = abs(float(row[t]))
        for k, value in enumerate(norms):
            size *= value ** int(pgd.exponents[t, k])
        # 1.0 unless `orthogonal_corrections` is on, where the deflated factor is
        # genuinely smaller than the power it replaces and the product above
        # would report the term the decomposition no longer holds.
        size *= pgd.deflation_shrinkage(m, t)
        corrections.append(size)
    return leading, corrections


def _print_term_magnitudes(pgd):
    """Print :func:`term_magnitudes` per mode, with each term's share of the mode.

    The share is what makes the table readable at a glance: the magnitudes span
    orders, so "is the correction doing anything" is a percentage question, not
    a magnitude one. ``lin`` is the leading (linear) term.
    """
    print()
    print("term magnitudes |coeff| * prod_j ||w_j||^lambda_j  (gauge-invariant, "
          "share of the mode)")
    for m in range(pgd.n_modes_truncated):
        leading, corrections = term_magnitudes(pgd, m)
        total = leading + sum(corrections)
        # A mode seeded but never trained can sit at exactly zero; dividing
        # would print nan for every share and hide that it is the *mode* that is
        # empty, which is itself the diagnosis.
        if total <= 0.0:
            print(f"  mode {m:2d}: (all terms zero -- this mode is empty)")
            continue
        cells = [f"lin {leading:10.3e} ({100.0 * leading / total:5.1f}%)"]
        for t in range(pgd.n_terms):
            lam = tuple(int(v) for v in pgd.exponents[t])
            cells.append(
                f"{lam} {corrections[t]:10.3e} "
                f"({100.0 * corrections[t] / total:5.1f}%)"
            )
        print(f"  mode {m:2d}: " + "   ".join(cells))


def _report(problem, history, verbose=True, plot=True):
    """Print the per-stage table and draw the figures for a finished run.

    Shared by the trained and the reloaded path, so a checkpoint produces
    exactly the same output as the run that wrote it -- otherwise the two paths
    drift and "plot from the checkpoint" stops being a faithful shortcut.

    Args:
        problem (Problem): the assembled objects, trained.
        history (TrainingHistory): the run to report on.
        verbose (bool): print the diagnostics table and the error breakdown.
        plot (bool): draw the three figures.
    """
    if verbose:
        print()
        print(f"training stopped: {history.stop_reason}")
        # `kind` names what each stage trained (see SCHEDULES); a run under one
        # of the CP baselines has one stage per mode and every row reads "cp".
        # `|C|` is the L1 norm of the stage's mode's coefficient row -- the only
        # column that looks at C at all, since `amplitude` and `max corr` are
        # computed from the monoms and are CP-only under every schedule. A run
        # whose `|C|` column is all zeros released nothing.
        # `c` is the mode's leading coefficient; blank under a decomposition that
        # pins it at 1, which is every run without `leading_coefficient`. Note it
        # moves under `renormalise` even when frozen -- the gauge fix parks the
        # mode's amplitude there -- so "c changed" is not by itself evidence that
        # the optimizer touched it. `mode` is printed because with a linear phase
        # the stage index is no longer the mode index.
        has_c = any("leading_coefficient" in r.diagnostics for r in history.stages)
        print(
            f"{'stage':>5} {'mode':>4} {'kind':>7} {'iters':>6} {'stop':>10} "
            f"{'energy':>14} {'gain':>12} {'amplitude':>11} {'max corr':>9} "
            f"{'|C|':>11}" + (f" {'c':>11}" if has_c else "")
        )
        for record in history.stages:
            kind = record.diagnostics.get("kind", "cp")
            mode = record.diagnostics.get("mode", record.stage)
            coefficient_norm = record.diagnostics.get("coefficient_norm", 0.0)
            row = (
                f"{record.stage:5d} {mode:4d} {kind:>7} {record.n_iter:6d} "
                f"{record.stop_reason:>10} "
                f"{record.energy:14.6e} {record.gain:12.4e} "
                f"{record.diagnostics['amplitude']:11.4e} "
                f"{record.diagnostics['max_correlation']:9.3f} "
                f"{coefficient_norm:11.4e}"
            )
            if has_c:
                c = record.diagnostics.get("leading_coefficient")
                row += f" {c:11.4e}" if c is not None else f" {'':>11}"
            print(row)

        if isinstance(problem.pgd, PolynomialNLPGD):
            # The coefficients are the whole point of this example, and no
            # inherited diagnostic looks at them: a row that stayed at zero means
            # the correction bought nothing (or was never released).
            print()
            print("polynomial coefficients C (one row per mode)")
            print("  exponents:", [tuple(int(v) for v in r) for r in problem.pgd.exponents])
            for m in range(problem.pgd.n_modes_truncated):
                row = problem.pgd.coefficients[m].detach()
                line = f"  mode {m:2d}: " + "  ".join(f"{v:11.4e}" for v in row)
                if problem.pgd.has_leading_coefficients:
                    c = float(problem.pgd.leading_coefficients[m].detach())
                    line += f"   (c = {c:11.4e})"
                print(line)

            _print_term_magnitudes(problem.pgd)

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


#: Selectable from the command line, so the schedules can be run back to back
#: against the same saved FEM reference: ``python <this file> simultaneous``.
#: ``polynomial`` is the only one that releases the coefficient rows; the other
#: two are the CP baseline, since a frozen ``C = 0`` reproduces CP-PGD exactly.
STRATEGIES = {
    "joint": JointNLGreedyTrainer,
    "staged": StagedNLGreedyTrainer,
    "refine": RefineNLGreedyTrainer,
    "support": SupportNLGreedyTrainer,
    "greedy": GreedyTrainer,
    "simultaneous": SimultaneousTrainer,
}

#: Which trainers accept the polynomial-only knobs (``n_linear_modes``,
#: ``leading_coefficient``, ``linear_stage_criterion``). The two library
#: trainers are the CP baseline and take none of them, so ``main`` must not
#: forward them -- hence a membership test rather than a ``try: except TypeError``
#: that would also swallow a real signature mistake.
POLYNOMIAL_TRAINERS = (PolynomialGreedyTrainer,)


def checkpoint_path(trainer_cls):
    """Default checkpoint file for a strategy: one file per strategy, so the two
    runs never overwrite each other and both stay available for plotting.

    Args:
        trainer_cls (type): the training strategy.

    Returns:
        Path: ``nlpgd5_<strategy>.pt`` in ``PARAM_SWEEP_DIR``.
    """
    return PARAM_SWEEP_DIR / f"nlpgd5_{trainer_cls.__name__.replace('Trainer', '').lower()}.pt"


#: Iterations per stage, per strategy. A simultaneous stage re-fits every
#: earlier mode on top of growing the new one, so it needs the longer budget --
#: at greedy's 120 its new modes never take off. Measured, not guessed, on the CP
#: example. The polynomial entry is **inherited from greedy, not measured**: its
#: CP sub-stage is a greedy stage, and how long the correction sub-stage needs is
#: exactly one of the things this example exists to find out.
STAGE_MIN_ITER = {
    JointNLGreedyTrainer: 120,
    StagedNLGreedyTrainer: 120,
    RefineNLGreedyTrainer: 120,
    SupportNLGreedyTrainer: 120,
    GreedyTrainer: 120,
    SimultaneousTrainer: 300,
}


if __name__ == "__main__":
    # python <this file> [joint|staged|refine|support|greedy|simultaneous] [--retrain]
    #
    # Without --retrain, an existing checkpoint is loaded and only the figures
    # are redrawn: changing a plot must not cost a training run.
    arguments = sys.argv[1:]
    retrain = "--retrain" in arguments
    positional = [a for a in arguments if not a.startswith("-")]
    name = positional[0] if positional else "joint"
    if name not in STRATEGIES:
        raise SystemExit(f"unknown strategy {name!r}; pick one of {sorted(STRATEGIES)}")
    print(f"training strategy: {name}")
    main(trainer_cls=STRATEGIES[name], retrain=retrain)
