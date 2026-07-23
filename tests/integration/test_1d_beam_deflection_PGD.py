import math

import pytest
import torch

from neurom.decompositions import Axis, CPPGD
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearBar
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Connectivity, Mesh
from neurom.fields import Field
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.interpolation.integration_domain import IntegrationDomain
from neurom.constraints import Dirichlet, NoConstraint
from neurom.math import jacobian
from neurom.math import inner
from neurom.math import integrate
from neurom.field_layout import FieldLayout

torch.set_default_dtype(torch.float32)


class Test1dBeamDeflection:
    """Integration test for the very first tensor-decomposition (CP-PGD) problem:
    a parametric 1D beam deflection u(x, E) whose analytical solution is rank-1
    separable,

        u(x, E) = 0.5 q (x - x_min)(x - x_max) / E,

    (a space parabola times 1/E). This test *bulletproofs* that first PGD by
    checking quantitative properties of the greedy enrichment rather than
    eyeballing plots:

      1. bounded error       -- final relative L2 error vs analytical < tol
      2. error decreases     -- rel. error is non-increasing as modes are added
      3. energy decreases     -- the minimised functional never goes up when the
                                 approximation space is enriched (core PGD prop.)
      4. no NaN/Inf          -- loss and error stay finite throughout
      5. Dirichlet BCs        -- u = 0 at the clamped ends for every E
      6. decreasing modes     -- each successive mode contributes less (spectral
                                 decay: the greedy PGD orders modes by importance)
      7. rank-1 recovery      -- the first mode alone already captures most of the
                                 field (the truth is rank-1), so it dominates the
                                 later corrective modes.

    The enrichment is driven by a *relative-energy plateau*: we keep training the
    active mode until the energy's relative improvement over a sliding window of
    iterations drops below ``epsilon``, then freeze it and add the next mode (see
    the stopping-parameters block below). ``epsilon`` is kept deliberately high so
    that several modes are enriched even though a single mode could in principle
    represent the separable truth -- this is what exercises the multi-mode
    machinery (cross-terms, freezing, per-mode optimiser groups).
    """

    # --- enrichment / stopping parameters (all tunable) --------------------
    # The mode is trained until the energy plateaus, measured as the *relative
    # improvement over a sliding window* of `plateau_window` iterations dropping
    # below `epsilon`. Two subtleties of this Adam optimisation drive the design:
    #   * the functional crosses zero (starts ~+2e5, ends ~-1e8) and spans ~8
    #     orders of magnitude, so we guard the relative denominator with a floor
    #     of 1.0 and use a window rather than a single-step difference;
    #   * Adam has a long *sticky* early phase (~100 iters where the energy barely
    #     moves and u is still ~its init) before it escapes and dives -- a plateau
    #     detector would mistake that for convergence, so `min_epochs_per_mode`
    #     forces every mode past it before the plateau test can fire.
    # `epsilon` is kept deliberately high so mode 0 stops well before full
    # convergence, leaving a genuine residual for modes 1 and 2 to capture (the
    # analytical field is rank-1, so a fully-converged mode 0 would starve them).
    epsilon: float = 2e-2  # relative-energy plateau -> add next mode
    plateau_window: int = 20  # window (iters) over which we measure it
    max_epochs_per_mode: int = 600  # safety cap so a mode always terminates
    min_epochs_per_mode: int = 120  # skip Adam's sticky early plateau

    final_error_tol: float = 1.5e-1  # (1) bounded final relative L2 error
    rank1_error_tol: float = 5e-1  # (7) error mode 0 alone must already beat
    error_slack: float = 5e-3  # (2) absolute slack on the non-increasing check
    energy_slack: float = 1e-3  # (3) relative slack on energy monotonicity
    dirichlet_tol: float = 1e-4  # (5) |u| at the clamped ends, absolute
    contribution_slack: float = 1e-2  # (6) relative slack on decreasing modes
    strict_error_tol: float = 3e-2  # (8) accuracy after a joint polish of all modes
    polish_epochs: int = 200  # joint-refinement iterations feeding (8)

    def test_beam(self):
        # Deterministic run so the calibrated thresholds are meaningful.
        torch.manual_seed(0)

        ## Axis
        # Shape function
        sf = LinearBar()
        # Quadrature strategy
        quad = TwoPoints1D()

        # Mapping from/to reference/physical coordinates: built per axis below,
        # since a mapping is bound to one mesh and must not be shared.

        # Prepare Field layout and fill it with actual fields
        field_layout = FieldLayout()

        ## Space
        # Dimensions
        x_min = 0.0
        x_max = 10.0
        N_space = 30

        # Generate vertices and connectivity
        x_array = torch.linspace(x_min, x_max, N_space).unsqueeze(-1)
        nodes_space = torch.arange(0, N_space)
        elements_space = torch.vstack(
            [torch.arange(0, N_space - 1), torch.arange(1, N_space)]
        ).T

        connectivity_space = Connectivity(nodes_space, elements_space)
        nodes_positions_space = Field(
            name=f"space_positions", connectivity=connectivity_space, values=x_array
        )

        # Initialize displacement values
        u_init = 0.5 * torch.ones(N_space, 1)

        mesh_space = Mesh(connectivity_space, nodes_positions_space)

        axis_space = Axis(
            name="space",
            mesh=mesh_space,
            sf=sf,
            mapping=IsoparametricMapping1D(sf, mesh_space),
            quad=quad,
            constraint=Dirichlet(
                nodes=[0, N_space - 1], values_imposed=torch.zeros(2, 1)
            ),
            init_values=u_init,
        )

        ## E
        # Dimensions
        E_min = 10.0
        E_max = 100.0
        N_E = 20

        # Generate vertices and connectivity
        E_array = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)
        nodes_E = torch.arange(0, N_E)
        elements_E = torch.vstack([torch.arange(0, N_E - 1), torch.arange(1, N_E)]).T

        connectivity_E = Connectivity(nodes_E, elements_E)
        nodes_positions_E = Field(
            name=f"E_positions", connectivity=connectivity_E, values=E_array
        )

        # Initialize E mode values
        E_init = 0.5 * torch.ones(N_E, 1)

        mesh_E = Mesh(connectivity_E, nodes_positions_E)

        axis_E = Axis(
            name="E",
            mesh=mesh_E,
            sf=sf,
            mapping=IsoparametricMapping1D(sf, mesh_E),
            quad=quad,
            constraint=NoConstraint(),
            init_values=E_init,
        )

        ## CP PGD object
        pgd_approx = CPPGD(
            axes=[axis_space, axis_E], n_modes_max=3, name="pgd", n_modes_ini=1
        )
        # print(pgd_approx.directory())

        ###### Define constant load.
        # The load is a *field* f(x), not a raw nodal vector: it has to be
        # sampled at the SAME quadrature points as u so that inner(f, u) aligns
        # (this is exactly what neurom.physics.LoadPotential does). We give it its
        # own nodal values on the space mesh, then interpolate it once on the
        # space axis's quadrature (same sf / quad / mapping / connectivity as u).
        # For a load that is constant in E, this is the single rank-1 spatial
        # factor f_0(x) of the separated source f(x, E) = f_0(x) ⊗ 1(E); the E
        # factor "1" is what the Gm = ∫ lmbda dE term below carries implicitly

        load_value = 1000.0  # x^2 ou une autre expression mathématique
        load_field = field_layout.add(
            Field(
                name="load",
                connectivity=connectivity_space,
                values=load_value * torch.ones(N_space, 1),
            )
        )
        context_f = axis_space.context  # le même context que la partie spatiale
        assembly_f = QuadratureAssembly(context_f, sf, load_field)

        # Construction of the shared domain for the whole problem
        domain = IntegrationDomain(
            [*pgd_approx.assemblies(), assembly_f]
        )  # do not forget * to unpack

        # Creer le modele
        model = NeuROMModel(
            field_layout=field_layout,
            decomposition=pgd_approx,
            integration_domain=domain,
            loss=lambda out: energy(out, pgd_approx, load_name="load"),
        )

        ## add training
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.1,
        )

        def closure():
            optimizer.zero_grad()
            out = model()
            # print(out)
            loss = model.loss(out)
            loss.backward(retain_graph=True)
            return loss

        ### Training: greedy enrichment driven by a relative-energy plateau.
        # Instead of adding a mode after a *fixed* number of epochs, we train the
        # active mode until the energy plateaus -- its relative improvement over the last `plateau_window` iterations falls below `epsilon` -- and only then freeze it and enrich with the next mode. The epoch cap guarantees termination even if the plateau is never reached.

        loss_history = []  # every iteration (for the no-NaN sweep)
        loss_per_mode = []  # energy at the end of each mode's training
        error_per_mode = []  # rel. L2 error vs analytical after each mode
        contribution_per_mode = []  # L2 norm of each individual mode u_m

        n_modes_target = pgd_approx.n_modes_max

        for mode_idx in range(n_modes_target):
            if mode_idx > 0:
                pgd_approx.freeze_mode(mode_idx - 1)
                pgd_approx.add_mode()  # active le mode suivant
                model.add_mode_to_optimizer(optimizer)

            mode_hist = []  # this mode's energy trajectory (for the plateau test)
            loss_val = math.nan
            for epoch in range(self.max_epochs_per_mode):
                loss_val = optimizer.step(closure).detach().item()
                loss_history.append(loss_val)
                mode_hist.append(loss_val)
                # Relative improvement over the last `plateau_window` iterations.
                # Denominator floored at 1.0 so the zero-crossing of the energy
                # doesn't blow it up; only tested after the sticky early phase.
                if (
                    epoch + 1 >= self.min_epochs_per_mode
                    and len(mode_hist) > self.plateau_window
                ):
                    past = mode_hist[-1 - self.plateau_window]
                    denom = max(abs(loss_val), abs(past), 1.0)
                    rel_improvement = (past - loss_val) / denom
                    if rel_improvement < self.epsilon:
                        break

            loss_per_mode.append(loss_val)
            error_per_mode.append(
                relative_l2_error(
                    model,
                    x_min=x_min,
                    x_max=x_max,
                    E_min=E_min,
                    E_max=E_max,
                    load_value=load_value,
                )
            )
            contribution_per_mode.append(
                mode_contribution(
                    pgd_approx,
                    mode_idx,
                    x_min=x_min,
                    x_max=x_max,
                    E_min=E_min,
                    E_max=E_max,
                )
            )

        n_modes = pgd_approx.n_modes_truncated

        # --- Polish: joint refinement of all modes, for a strict accuracy check.
        # The greedy loop above deliberately under-trains each mode (high epsilon)
        # to exercise the enrichment machinery, so its final error is only loosely
        # bounded. Here we unfreeze every mode and refine them jointly: with the
        # full rank-N space trained to convergence the PGD must match the
        # analytical field *tightly*. This recovers the strict correctness
        # guarantee (the accuracy check the previous, LBFGS-based test asserted).
        for m in range(n_modes):
            pgd_approx.unfreeze_mode(m)
        polish_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.1,
        )

        def polish_closure():
            polish_optimizer.zero_grad()
            out = model()
            loss = model.loss(out)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(self.polish_epochs):
            polish_optimizer.step(polish_closure)

        final_error_polished = relative_l2_error(
            model,
            x_min=x_min,
            x_max=x_max,
            E_min=E_min,
            E_max=E_max,
            load_value=load_value,
        )

        print("Successfully trained!")
        print(f"modes enriched      : {n_modes}")
        print(f"energy per mode     : {loss_per_mode}")
        print(f"rel. error per mode : {error_per_mode}")
        print(f"contribution per mode: {contribution_per_mode}")
        print(f"error after polish  : {final_error_polished}")

        # ================================================================
        # Metrics / assertions -- this is what makes the test a test.
        # ================================================================

        # (4) No NaN/Inf anywhere -- catch a diverging optimisation early.
        assert all(math.isfinite(l) for l in loss_history), (
            "non-finite loss encountered"
        )
        assert all(math.isfinite(e) for e in error_per_mode), (
            "non-finite error encountered"
        )
        assert all(math.isfinite(c) for c in contribution_per_mode)

        # (1) Bounded error: the converged rank-N PGD matches the analytical
        #     parametric field over the whole (x, E) domain.
        assert error_per_mode[-1] < self.final_error_tol, (
            f"final relative L2 error {error_per_mode[-1]:.3e} exceeds "
            f"tolerance {self.final_error_tol:.3e}"
        )

        # (2) Error decreases (non-increasing) as modes are added. A small slack
        #     absorbs the fact that a converged higher mode may leave the error
        #     essentially unchanged rather than strictly lower.
        for i in range(len(error_per_mode) - 1):
            assert error_per_mode[i + 1] <= error_per_mode[i] + self.error_slack, (
                f"error increased when adding mode {i + 1}: "
                f"{error_per_mode[i]:.3e} -> {error_per_mode[i + 1]:.3e}"
            )

        # (3) Energy monotonically decreases under enrichment: adding a mode
        #     enlarges the trial space, so the minimum of the functional can only
        #     drop (or stay). Slack is relative to the current energy magnitude.
        for i in range(len(loss_per_mode) - 1):
            tol = self.energy_slack * (abs(loss_per_mode[i]) + 1e-12)
            assert loss_per_mode[i + 1] <= loss_per_mode[i] + tol, (
                f"energy increased when adding mode {i + 1}: "
                f"{loss_per_mode[i]:.6e} -> {loss_per_mode[i + 1]:.6e}"
            )

        # (5) Dirichlet BCs: u(x_min, E) = u(x_max, E) = 0 for every E. The ends
        #     are exact mesh nodes whose values are imposed, so this is tight.
        #     model(coords) only returns a pointwise tensor in eval mode (in train
        #     mode it returns the assembled FieldLayout used to build the energy).
        model.eval()
        E_probe = torch.linspace(E_min, E_max, 5)
        with torch.no_grad():
            u_lo = model(
                torch.stack([torch.full_like(E_probe, x_min), E_probe], dim=1)
            ).reshape(-1)
            u_hi = model(
                torch.stack([torch.full_like(E_probe, x_max), E_probe], dim=1)
            ).reshape(-1)
        assert u_lo.abs().max().item() < self.dirichlet_tol, (
            "Dirichlet BC violated at x_min"
        )
        assert u_hi.abs().max().item() < self.dirichlet_tol, (
            "Dirichlet BC violated at x_max"
        )

        # (6) Decreasing mode contribution: the greedy PGD peels off the most
        #     energetic content first, so ||u_m|| is non-increasing in m.
        for i in range(len(contribution_per_mode) - 1):
            slack = self.contribution_slack * (contribution_per_mode[0] + 1e-12)
            assert contribution_per_mode[i + 1] <= contribution_per_mode[i] + slack, (
                f"mode {i + 1} contributes more than mode {i}: "
                f"{contribution_per_mode[i]:.3e} -> {contribution_per_mode[i + 1]:.3e}"
            )

        # (7) Rank-1 recovery: the truth is separable, so the first mode alone
        #     already captures most of the field (small error, dominant norm) and
        #     the later modes are only small corrections.
        assert error_per_mode[0] < self.rank1_error_tol, (
            f"mode 0 alone should already approximate the rank-1 field, "
            f"got relative error {error_per_mode[0]:.3e}"
        )
        assert contribution_per_mode[0] > contribution_per_mode[1], (
            "the first mode should dominate the corrective modes"
        )

        # (8) Strict accuracy: once every mode is unfrozen and refined jointly to
        #     convergence, the full rank-N PGD reproduces the analytical field
        #     tightly -- the strict correctness guarantee, distinct from the loose
        #     bound (1) that the deliberately under-trained greedy pass gives.
        assert final_error_polished < self.strict_error_tol, (
            f"polished relative L2 error {final_error_polished:.3e} exceeds "
            f"strict tolerance {self.strict_error_tol:.3e}"
        )


## Metric helpers ----------------------------------------------------------


def analytical(x, E, *, x_min, x_max, load_value):
    """Analytical parametric beam deflection u(x, E) = 0.5 q (x-x_min)(x-x_max)/E.

    Broadcasts over any shapes of ``x`` and ``E`` (e.g. column x and row E give
    the full (N_x, N_E) grid).
    """
    return 0.5 * load_value * (x - x_min) * (x - x_max) / E


def relative_l2_error(model, *, x_min, x_max, E_min, E_max, load_value, n_x=60, n_E=40):
    """Relative L2 error of the assembled PGD field against the analytical one.

    Uses ``model.assemble`` to evaluate the full separated tensor u(x, E) on a
    regular (n_x, n_E) grid, then compares it to the analytical parametric
    solution in a discrete L2 (Frobenius) sense.
    """
    x_grid = torch.linspace(x_min, x_max, n_x)
    E_grid = torch.linspace(E_min, E_max, n_E)
    with torch.no_grad():
        u_pgd = model.assemble([x_grid, E_grid])  # (n_x, n_E)
    u_ana = analytical(
        x_grid.unsqueeze(-1),
        E_grid.unsqueeze(0),
        x_min=x_min,
        x_max=x_max,
        load_value=load_value,
    )
    return (torch.norm(u_pgd - u_ana) / torch.norm(u_ana)).item()


def _factor(pgd_approx, m, k, pts):
    """Evaluate the single monom w_m^k(.) of mode ``m`` on axis ``k`` at ``pts``.

    Same tool the decomposition uses internally: PointWiseInterpolator evaluates
    one monom field on its own axis mesh at arbitrary query points.
    """
    pwi = PointWiseInterpolator(
        pgd_approx.axes[k].mesh,
        pgd_approx.axes[k].sf,
        pgd_approx.monoms[m][k],
        pgd_approx.axes[k].mapping,
    )
    return pwi.at_position(pts.reshape(-1, 1, 1)).reshape(-1)


def mode_contribution(pgd_approx, m, *, x_min, x_max, E_min, E_max, n=200):
    """Discrete L2 norm over the domain of the single mode u_m = w_m^x ⊗ w_m^E.

    This is a scale-invariant proxy for how much mode ``m`` adds to the solution
    (the product w_m^x * w_m^E is unambiguous even though each factor carries an
    arbitrary scale individually).
    """
    x_grid = torch.linspace(x_min, x_max, n)
    E_grid = torch.linspace(E_min, E_max, n)
    with torch.no_grad():
        w_x = _factor(pgd_approx, m, 0, x_grid)
        w_E = _factor(pgd_approx, m, 1, E_grid)
    u_m = torch.outer(w_x, w_E)  # (n, n)
    return torch.sqrt(torch.mean(u_m**2)).item()


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
    E_val = [E_mode_field.x.values for E_mode_field in E_modes]
    lmbdas = [E_mode_field.u.values for E_mode_field in E_modes]
    J_E = [E_mode_field.measure.values for E_mode_field in E_modes]

    # space part
    u = [space_mode_field.u.values for space_mode_field in space_modes]
    x_val = [space_mode_field.x.values for space_mode_field in space_modes]
    # jacobian returns (N_e, N_q, *u_shape, d): one extra trailing axis of
    # size d (the physical dimension) compared to u's own shape (N_e, N_q,
    # *u_shape). That axis must be *contracted away* -- the elastic term is the
    # scalar product grad(u_m) . grad(u_n) summed over the spatial directions --
    # so we keep the raw jacobian output here (no reshape) and let inner()
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
    grad_u = [jacobian(x_val[n], u[n]) for n in range(n_modes)]
    J_u = [space_mode_field.measure.values for space_mode_field in space_modes]

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
    load_f = load_field.u.values
    for m in range(n_modes):
        Fx = integrate(inner(load_f, u[m]) * J_u[m])
        Gm = integrate(lmbdas[m] * J_E[m])
        load = load + Fx * Gm

    return elastic + load


if __name__ == "__main__":
    Test1dBeamDeflection().test_beam()
