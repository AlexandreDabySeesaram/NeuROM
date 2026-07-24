## 2026-07-24 — FEM reference solutions for the 5-parametric bar, and plotting against them

- **Added** `docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py`
  and the bundle it writes, `reference_solution.pt` (12 kB). One plain
  `FEMModel` solve per parameter point — 400 nodes, `TwoPoints1D`, LBFGS — at 10
  seeded-uniform `(E1, E2, alpha, n)` points, sampled on a fixed 101-point `x`
  grid. Saving it is the point: every decomposition strategy is scored against
  *the same* numbers without re-solving. Regenerate with
  `python docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py`
  (~20 s); the parameter points are seeded, so they stay comparable across runs.
- **Library change**: `ElasticEnergy(field, modulus=None)` now accepts an
  optional modulus — either a field looked up in the layout (like
  `LoadPotential`'s load) or a callable evaluated at the quadrature points.
  Default `None` keeps the old unit-modulus behaviour; every existing test
  passes untouched. The reference passes the modulus as a *callable*, so `E(x)`
  is exact at the quadrature points instead of piecewise-linearly interpolated.
- **The reference runs in float64** (scoped `double_precision()` context, cast
  back to the caller's dtype on load). In float32 it stalled at ~1e-3 relative:
  halving both moduli, which must scale `u` by exactly 2, was off by 3e-3, and
  the constant-modulus check sat at 2.7e-4. In float64 those are 8e-5 and
  **6.3e-6** — the reference is no longer the limiting error.
- **Discarded approach**: a first version computed the reference analytically by
  quadrature (`(E u')' = f` → `u = int (C + f s)/E(s) ds`). It was correct
  (6e-5 vs the constant-`E` parabola) but it is not what the surrogate is
  approximating; the direct FEM solve is the right yardstick and is reusable.
- **Added** plotting to the example: `plot_convergence` (energy vs iteration
  with stage boundaries), `plot_solution` (one panel per reference parameter
  point, PGD vs FEM, per-panel relative L2), `plot_modes` (each mode's per-axis
  factor, normalised — CP factors are only defined up to a per-axis scale, so
  coinciding curves are the degeneracy `max_correlation` reports). `main` gained
  `plot=True`; matplotlib is imported lazily inside the helpers.
- **Result**: a 3-mode greedy CP-PGD (default 30/20/20/15/15 nodes,
  `RelativeChange(tol=1e-3, window=20, max_iter=400, min_iter=120)`) reaches
  **4.2 % mean / 13.8 % worst** relative L2 error over the 10 reference points.
  The worst point is the stiffest one (`E1=97, E2=74`), where the PGD
  under-deflects; the softer points are at 1–2 %.

## 2026-07-24 — GreedyTrainer wired into the 2-parametric example; validated against the analytical solution

- **Changed** `docs/examples/1d_2-parametric_beam_PGD/1d_beam_deflection_PGD.py`:
  the three copy-pasted 150-iteration blocks (`freeze_mode` / `add_mode` /
  `add_mode_to_optimizer`, one persistent `Adam` + closure reused across all
  three) are gone, replaced by a single `GreedyTrainer(...).enrich()` call with
  `DEFAULT_STAGE_CRITERION` (`RelativeChange(tol=1e-3, window=20, max_iter=600,
  min_iter=120)`) and `DEFAULT_ENRICHMENT_CRITERION` (`RelativeGain(tol=1e-3)`).
  Setup moved into `build_problem(loss_fn, *, n_modes_max=3, n_modes_ini=1,
  n_nodes=None, quad=None) -> Problem`, mirroring the 5-parametric example's
  `make_axis` / `Problem` / `build_problem` shape exactly (this example is now
  structurally parallel to it). `Problem` carries `x_min`, `x_max`, `E_min`,
  `E_max`, `load_value` alongside the usual `model`/`pgd`/`field_layout`/
  `domain`/`axes`/`history`, since this is the one problem whose exact
  analytical solution can be checked against those scalars.
- **This is the only task in the plan that checks the answer, not the
  machinery.** The 2-parametric beam is exactly rank-1
  (`u = 0.5 q (x-x_min)(x-x_max) / E`). New test
  `tests/integration/test_greedy_trainer.py::test_greedy_trainer_recovers_the_analytical_beam`
  trains via `GreedyTrainer` and asserts the trained model's relative L2 error
  against that closed form (dense 60x40 (x, E) grid) is below
  `ANALYTICAL_ERROR_TOL`.
- **Measured relative L2 error: 0.1019** (10.2%), identical across
  `manual_seed(0)` and `manual_seed(42)` — this pipeline has no randomness
  beyond the deterministic `0.5*ones` monom seed, so the run is fully
  reproducible, not a lucky draw. `ANALYTICAL_ERROR_TOL` set to `0.21`, roughly
  double the measurement, rounded up.
- **Correction to the comparison below, and to an earlier version of this
  entry: `final_error_tol` (15%) and `strict_error_tol` (3%) in
  `tests/integration/test_1d_beam_deflection_PGD.py` are *tolerances*, not
  *measurements* — comparing `GreedyTrainer`'s 0.1019 against them and calling
  it "better" or "no regression" was wrong.** Rerunning that baseline test with
  `-s` gives the actual numbers: `rel. error per mode : [0.3269, 0.1807,
  0.09536]`, `error after polish : 0.01527`. So the hand-rolled loop's real
  greedy-regime error is **0.0954**, not 15%, and `GreedyTrainer`'s **0.1019**
  is **marginally worse**, by about 7% relative. Note the comparison is not
  against the old example's fixed-150-iteration loop — no measurement here
  comes from that. It is against the *baseline test*, which runs its own
  hand-calibrated plateau rule (`epsilon=2e-2`, `plateau_window=20`,
  `min_epochs_per_mode=120`, `max_epochs_per_mode=600`). That rule and
  `RelativeChange(tol=1e-3, window=20, min_iter=120, max_iter=600)` are nearly
  the same criterion; they differ mainly in `tol` (1e-3 vs 2e-2), which makes
  the two numbers a fair like-for-like comparison rather than a confound.
  The polished baseline is **0.0153**, not 3% — an extra all-modes joint
  optimization stage `GreedyTrainer` does not perform, so the trainer does not
  yet close that gap either.
- **More interesting than the 7%: the trainer's stage 0 converges much
  better than the baseline's** (mode-0-alone error **0.1305** vs. the
  baseline's **0.3269**), yet its final rank-3 result is still slightly worse.
  On a truth that is exactly rank-1, a better mode 0 leaves less real signal
  in the residual for modes 1-2 to fit — they are left fitting numerical
  residue either way, so a stronger mode 0 does not guarantee a stronger
  rank-3 sum. A future `simultaneous`/`greedy+update` strategy — with an
  all-modes polish stage — is the natural place to close both gaps.
- **What that test does and does not discriminate**, measured rather than
  assumed: an untrained model scores `1.0006` and an undertrained one (<=50
  iterations) `~1.0000`, so `0.21` is a real bound, not a vacuous one. But mode
  0 **alone** scores `0.130`, also under the bound — the truth being exactly
  rank-1, mode 0 is most of the answer. So this test mainly exercises stage-0
  convergence; the enrichment machinery is pinned separately by
  `test_max_correlation_is_one_for_a_deliberately_duplicated_mode`. The
  answer-check and the enrichment-check are not the same check here.
  Relatedly, `torch.manual_seed(...)` pins nothing at present: nothing under
  `src/neurom/` draws random numbers, and monoms start from a deterministic
  `0.5*ones`.
- **Trained `max_correlation`: stage 0 = 0.000 (nothing to correlate against),
  stage 1 = 0.242, stage 2 = 0.161.** Both well below 1.0 — modes 1 and 2 do
  *not* duplicate mode 0 here, unlike the 5-parametric problem at short stage
  lengths (Task 3's negative result). Training stopped at `"capacity"`
  (`n_modes_max=3` reached) rather than `RelativeGain` converging first, so a
  larger mode budget was not tried; with the true rank being 1, modes 1 and 2
  are pure numerical residue and their shrinking amplitudes (7.04e3, 1.34e3,
  4.17e2) and gains (stage 1 gain 2.46e5, stage 2 gain 3.98e3) are consistent
  with that, not with degeneracy.
- Full suite: 157 passed (156 baseline + 1). Full report:
  `.superpowers/sdd/task-5-report.md`.

## 2026-07-24 — GreedyTrainer wired into the 5-parametric example

- **Changed** `docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
  `main(verbose=True, train=True)` now trains after the untrained energy is
  printed, via `GreedyTrainer` with `RelativeChange(tol=1e-3, window=20,
  max_iter=600, min_iter=120)` and `RelativeGain(tol=1e-3)`. `Problem` gained a
  `history` field (`None` unless trained). `train=False` reproduces the old,
  untrained behaviour so the pre-existing energy-matches-printed-value test
  keeps its exact assertions.
- **Verified** (`tests/integration/test_1d_5_parametric_beam_energy.py`, +1
  test, 18 total): greedy enrichment gets past stage 0 (`len(stages) > 1` and
  `n_modes_truncated > 1`), every later stage's energy is at or below stage 0's,
  the concatenated losses stay finite, and the table is printed. Note
  `n_modes_truncated == len(history.stages)` is kept as a consistency check but
  carries no information on its own — it is an architectural tautology of
  `GreedyTrainer` at `n_modes_ini=1`, true whether or not a mode is ever added.
- **Result — `min_iter=120` keeps modes distinct on the real (non-tiny) 5-parametric
  problem, at least in this one run.** All 8 stages before `RelativeGain`
  stopped the run converged in exactly 120 iterations (the criterion's first
  eligible check, at `min_iter`, already cleared `tol` every time — consistent
  with the "Adam sticky early phase" this criterion was designed around).
  `max_correlation` per stage: `0.000, 0.086, 0.159, 0.114, 0.145, 0.198,
  0.433, 0.086` — well below the 1.0 seen at `FixedIterations(15)` in Task 3,
  confirming that negative result and its fix on the full-size mesh, not just
  the tiny test mesh. Energy fell monotonically from `-1.905e11` to
  `-2.045e11` across the 8 stages; gains shrank from `5.6e9` to `9.1e7`, and
  `RelativeGain(tol=1e-3)` then stopped enrichment at stage 8 with reason
  `"converged"` (capacity, `n_modes_max=10`, was not reached). Single run, not
  ablated across seeds — stage 6's jump to `max_correlation=0.433` is the
  largest value seen and worth watching if this is rerun.
- **`min_iter` is load-bearing, and this was ablated.** Replaying the same run
  with `min_iter=1`: stages 1, 3, 4, 5, 6 and 7 all falsely fire `"converged"`
  at iteration 21. Tracing stage 1, the relative change dips to `3.8e-5` at
  n=21 (a false plateau), climbs back to `0.023` by n=50 as Adam escapes and
  the loss dives from `-1.9046e11` to `-1.9593e11`, then decays back under
  `tol` around n=80. That is the "Adam sticky early phase" the criterion was
  designed around, reproduced concretely. Conversely `tol` is not decorative
  either: extending stage 1 to n=300 and stage 6 to n=400 shows the relative
  change shrinking to 1e-6–1e-7 and stage 6's `max_correlation` drifting only
  0.433 -> 0.423, so stopping at 120 is genuine convergence detection rather
  than a disguised `FixedIterations(120)`.
- Full suite: 156 passed (155 baseline + 1). Full report:
  `.superpowers/sdd/task-4-report.md`.

## 2026-07-23 — PGDTrainer: a base class for interchangeable PGD training strategies

- **New module `src/neurom/training/`.** `PGDTrainer` (ABC, `base.py`) owns
  the loop `enrich()` -> `stage()` -> `step()`: a run is a sequence of stages,
  each iterating `step()` until its stage criterion fires, until an
  enrichment criterion (or the strategy) says to stop adding stages.
  `history.py` holds the plain-data records, `StageRecord` (one stage's
  losses, stop reason, end-of-stage energy, gain, diagnostics) and
  `TrainingHistory` (the ordered list plus a run-level stop reason).
  `criteria.py` holds two protocols -- `StageCriterion` (watches a stream of
  per-iteration losses) and `EnrichmentCriterion` (watches completed stages)
  -- and four concrete criteria: `RelativeChange`, `FixedIterations`,
  `RelativeGain`, `MaxStages`. `GreedyTrainer` (recorded above, 2026-07-23) is
  the first concrete strategy built on this base.
- **The base deliberately knows nothing about modes.** It never calls
  `add_mode` or `mode_parameters`, and never assumes a stage index is a mode
  index; the trainable set for each stage is read off
  `[p for p in model.parameters() if p.requires_grad]`. All decomposition
  manipulation -- freezing, adding a mode, building the optimizer -- lives in
  the `prepare_stage` hook that subclasses implement. This is what is meant to
  let the same loop drive the planned non-linear decompositions, whose
  trainable terms are not all modes.
- **Criteria are injected objects, not overridden methods.** Overriding would
  need a subclass per (strategy x stopping rule) pair
  (`GreedyWithRelTol`, `GreedyWithFixedIter`, `AlternatingWithRelTol`, ...);
  injecting `stage_criterion`/`enrichment_criterion` keeps the strategy and
  the stopping rule independent. Both return a reason string rather than a
  bool, so `TrainingHistory`/`StageRecord` record *why* a run or stage ended
  (`"converged"`, `"max_iter"`, `"n_iter"`, `"capacity"`, `"diverged"`), not
  just that it did.
- **`step()` is closure-based** (`self.optimizer.step(closure)`) so LBFGS
  works with no branching, and it returns the loss *before* that iteration's
  update -- that is the contract of `optimizer.step(closure)`. That is why
  `StageRecord.energy` prefers a separately recorded `final_energy`
  (`_record_final_energy`, one extra forward pass after the stage's criterion
  fires) over `losses[-1]`: using the last recorded loss as "the stage's
  energy" would put a one-update error into every `gain` and every
  enrichment decision.
- **`RelativeChange` needs `min_iter` and a denominator floor of 1.0**, both
  carried over unchanged from the criterion calibrated by hand in
  `tests/integration/test_1d_beam_deflection_PGD.py`. Without `min_iter`, Adam's
  long sticky early phase on this problem (energy barely moves for ~100
  iterations before it escapes and dives) reads as a plateau and stops mode 0
  after ~20 iterations. Without the floor, the energy crossing zero
  (~+2e5 to ~-1e8) makes a purely relative denominator blow up near the
  crossing and the stage never stops.
- **Verified**: 34 unit tests in `tests/unit/training/` (`test_base.py`,
  `test_criteria.py`, `test_history.py`), against a synthetic model/loss, not
  real physics. Cover: loop monotonicity, frozen parameters left bitwise
  unchanged, every stop reason firing (`converged`, `max_iter`, `n_iter`,
  `capacity`, `diverged`), a diverging stage stopping the whole run rather
  than poisoning later stages, and `enrich()` being resumable (calling it
  again continues from `len(history.stages)`).
- **Independent confirmation from the final review's own experiment**: writing
  a `SimultaneousTrainer` against this API needed only `prepare_stage` and
  `should_add_stage` overrides, confirming the "the base knows nothing about
  modes" design claim for that family of strategies. But an
  `AlternatingTrainer` needs two seams that do not exist yet -- see "Not yet
  done" below.
- **Not yet done:**
  - `CPPGD` exposes `freeze_all`/`freeze_mode`/`unfreeze_mode` but nothing
    per-monom, so an alternating-directions strategy (sweeping one axis of one
    mode at a time) has to reach past the public API into
    `monoms[m][k].values_reduced` directly. A `freeze_monom(m, k)` /
    `unfreeze_monom(m, k)` pair on `CPPGD` is the missing seam.
  - `on_stage_end` runs *before* `history.append`, so inside that hook
    `record.gain` is still NaN -- it is only filled in by
    `TrainingHistory.append` afterwards. A strategy wanting the gain inside
    `on_stage_end` currently cannot get it there.
  - **Divergence is detected but not latched.** `enrich()` breaks out of the
    loop on a diverged stage, but calling `enrich()` again resumes on the
    poisoned state and burns another mode; `history.stop_reason` is silently
    overwritten. `RelativeGain` cannot stop it either -- `(previous - current)
    / denom < tol` is False under NaN, so a NaN history reads as "still
    gaining". Self-limiting in practice (the poisoned state re-diverges at
    once, so it is not data corruption), but `enrich()` should refuse to
    resume a history whose last stage diverged.

## 2026-07-23 — GreedyTrainer, and two base.py bugs only real physics exposed

- **New** `src/neurom/training/greedy.py`: `GreedyTrainer(PGDTrainer)`, the
  first concrete strategy. One stage = one mode: stage 0 trains the
  initially-active mode(s), every later stage calls `freeze_all()` then
  `add_mode()` before building a fresh optimizer. `should_add_stage` checks
  `n_modes_truncated >= n_modes_max` itself (stop reason `"capacity"`) so
  `CPPGD.add_mode()`'s `RuntimeError` at capacity is never hit. `on_stage_end`
  records two diagnostics per stage from raw nodal vectors (cheap, no forward
  pass): `amplitude` (`prod_k ||w_m^k||`) and `max_correlation` (largest
  normalised rank-1 inner product against earlier modes) — meant to *spot*, not
  quantify, a greedy step rediscovering an existing mode (every mode shares the
  same `Axis.init_values` seed).
- **Verified** (`tests/integration/test_greedy_trainer.py`, 7 tests, against
  the real 5-parametric beam on a tiny mesh
  `{space:5, E1:4, E2:6, alpha:7, n:3}`, `n_modes_max=3`): one mode added per
  stage, stage 0 does not enrich, frozen modes are bitwise unchanged by later
  stages, capacity stops the run before `add_mode()` can raise, and the first
  mode's `max_correlation` is exactly 0. The end-of-stage-energy-never-rises
  property (`FixedIterations(80)`, tolerance `1e-9 * |previous.energy|`)
  **passed unmodified** — 80 iterations was enough for each stage to work off
  the seam jump from the new mode's nonzero seed. Note the caveat below: at
  the shorter `FixedIterations(15)` most tests use, the diagnostics being "in
  range" hides that they are pinned at a degenerate value, not a healthy one
  — see the negative result below.
- **Two pre-existing `base.py` bugs found and fixed**, both invisible to Task
  2's synthetic stub loss and both hit by every single new test on first run:
  1. `step()`'s `loss.backward()` needed `retain_graph=True`. Cause:
     `QuadratureContext.__init__` builds `x_phys` **and** `xi_back` once, at
     construction, from the `requires_grad` leaf `_xi_ref`
     (`_compute_quad_pos`); `NeuROMModel.forward` only calls
     `interpolate_all` and never `IntegrationDomain.update_contexts()`, so
     every iteration's forward reads through that same already-built
     subgraph. A non-retaining `backward()` frees it after the first
     iteration, so the second iteration's backward raises "Trying to backward
     through the graph a second time". This is **not** about
     `jacobian_field`'s `create_graph=True`: an energy with no
     `jacobian_field`, no `create_graph`, and no reference to `_xi_ref` fails
     identically on iteration 2, because it still reads interpolated
     quantities built on that one shared subgraph. Confirmed by reproducing
     standalone with a bare `Adam` + closure (2 steps, no trainer). Expected
     to bite *any* strategy trained against *any* energy that reads
     interpolated fields, not just `GreedyTrainer` or autograd-differentiated
     ones.
  2. `_record_final_energy` wrapped its forward pass in `torch.no_grad()`,
     which strips `grad_fn` from every intermediate tensor regardless of the
     underlying leaves — fatal for any energy using `jacobian_field` internally
     (`torch.autograd.grad` has nothing to differentiate). Fixed by dropping
     `no_grad()` and using `.detach()` on the returned loss instead, matching
     the pre-trainer 2-parameter example's own `loss.detach().item()` pattern.
     Not anticipated by the Task 3 brief; a latent defect in already-committed
     code that would otherwise have broken end-of-stage energy tracking
     (`StageRecord.energy`, `gain`, `RelativeGain`) for every future strategy
     trained against real physics.
- **Negative result — short stages make greedy enrichment degenerate.** On the
  tiny-mesh 5-parametric problem the new `max_correlation` diagnostic
  immediately earned its keep. At `FixedIterations(15)` (what most of the tests
  use) it reads **1.0** for stages 1 and 2, amplitudes match to 4 decimals, and
  the stage energies come out exactly 1x/2x/3x — three copies of the same mode.
  Measured sweep of (energies, correlations): 5 iters -> energies RISE; 15 ->
  corr 0.0/1.0/1.0; 40 -> corr 0.0/0.999998/0.999996; 80 -> corr
  0.0/0.0092/0.176; 200 -> corr 0.0/0.066/0.135. So ~80 iterations per stage is
  where the modes actually separate on this problem. **Do not read a falling
  energy as successful enrichment** — check `max_correlation`.
  *Cause not established.* The plausible story is that every mode carries the
  same `0.5*ones` seed and that early on the linear load term dominates, so a
  fresh mode retraces mode 0's trajectory — but that was never ablated, only
  the duplication itself was measured (the 15-iteration row was reproduced
  independently; the other rows are from a single run). Whether a per-mode
  seeding strategy is needed is still open, and so is the mechanism.
  *Later, concrete evidence for the seed-symmetry mechanism*: the final review
  measured that at `n_modes_ini=2`, stage 0 reports `max_correlation = 1.0`
  with amplitudes matching to 4 significant digits — every initially-active
  mode shares the same `Axis.init_values` seed and sees the same gradient, so
  joint training keeps them parallel forever. This is the one case where the
  mechanism is forced rather than merely plausible, since there is no freezing
  between the two modes to even hypothetically break the symmetry.
- Full suite: 155 passed (146 baseline + 9), no regressions. Full report:
  `.superpowers/sdd/task-3-report.md`.

## 2026-07-22 — 5-parametric beam with tanh-graded modulus

- **New example** `docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
  the 1D bar now has a two-zone Young's modulus
  `E(x, E1, E2, alpha, n) = (E2-E1)/2 tanh(n(x-alpha)) + (E2+E1)/2`, giving five
  coordinates `(x, E1, E2, alpha, n)` and a CP-PGD
  `u = sum_i X_i lambda_i mu_i A_i N_i`.
- **Architecture:** the energy is *injected* into `build_problem(loss_fn, ...)`, so the
  same wiring will drive the non-linear PGD functionals unchanged. Nothing added under
  `src/` — the energy stays inline while the formulation moves. The existing
  `physics.Term` contract is single-axis (`integrand -> (N_e, N_q, 1)`) and does not fit
  a multi-axis separated energy; extending it is deliberately deferred.
- **Key numerical point:** `tanh(n(x-alpha))` is *not* separable — it couples x, alpha
  and n. It is integrated by an exact 3-D tensor-product quadrature over those axes'
  quadrature points (one `einsum` per mode pair, grid `(29, 14, 14)`, rebuilt each call
  so r-adaptive meshes stay correct). The `(E2 +/- E1)/2` prefactors remain 1-D moments.
  Rejected: offline low-rank separation of `tanh` (extra approximation error) and
  Monte-Carlo sampling (noisy gradients).
- **Verified** (`tests/integration/test_1d_5_parametric_beam_energy.py`, 17 tests): the
  separated energy matches a brute-force 5-D tensor-product quadrature — one that
  exploits no separability at all — to 1e-9 in double precision, for 1 and 2 modes and
  for both `MidPoint1D` and `TwoPoints1D`; it is differentiable w.r.t. every monom of
  every active mode; and it still matches the reference after `add_mode()`, pinning the
  greedy-enrichment seam. Two deliberate-bug injections confirmed the tests bite (a
  wrong-coordinate-axis read, and a corrupted printed value). At the 0.5 seed the
  script prints `energy = 6.602113e+07`, which an independent hand computation
  reproduces to 5 significant figures.
- **`build_problem` takes an injectable `quad`** (default `MidPoint1D`). This exists to
  make the `N_q > 1` path testable: one quadrature point per element previously hid a
  broadcasting bug in the 2-parameter ancestor that was silently wrong at `N_q=1` and
  crashed at `N_q=2`. Test-authoring trap found here: instantiating a quadrature rule at
  `@parametrize` decoration time freezes its buffers at collection-time float32 and
  clashes with a float64 fixture — parametrize over the class, instantiate in the body.
- **Not yet done:** training, plotting, and a reference solution. Mode budget set to 10
  on the assumption that the non-separable modulus is far from rank-1 — untested.
- Parameter ranges: `x in [0,10]` (30 nodes), `E1, E2 in [10,100]` (20),
  `alpha in [2,8]` (15), `n in [0.5,5]` (15). Note `n` never reaches 0, so the
  near-uniform-modulus regime is not sampled.
- **Known scaling limit (first thing that will break):** the `(Qx, Qalpha, Qn)` tanh grid
  is built *inside* the autograd graph and rebuilt every iteration. At the defaults that
  is 5 684 entries — trivial — but ~300 space elements with ~100 each in alpha and n
  gives a 3 M-entry graph-retained tensor per iteration. Cache it (invalidating on mesh
  change) or contract alpha/n first, and exploit the `(i,j)`/`(j,i)` symmetry the
  mode-pair loop currently ignores. Related trap: the grid hard-codes mode 0's
  quadrature points, so per-mode independent meshes would corrupt the energy *silently*
  rather than crash.
