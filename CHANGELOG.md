## 2026-07-29 — `PolynomialNLPGD`: first non-linear PGD decomposition

**State:** new `src/neurom/decompositions/polynomial_pgd.py` +
`tests/unit/decompositions/test_polynomial_pgd.py` (28 tests, all pass).
Exported from `decompositions/__init__.py`. **No example uses it yet** — the
5-parametric beam still runs `CPPGD`; rewiring its `energy()` to consume
`polynomial_directory()` is the missing piece. The 2 pre-existing failures in
`test_1d_5_parametric_beam_energy.py` (stale checkpoint, modes 10–11) are
unchanged by this work.

- `u = sum_i ( prod_j w_ij + sum_{lambda in I} C_{i lambda} prod_j w_ij^lambda_j )`.
  Subclasses `CPPGD`: monom grid, assemblies and greedy lifecycle inherited
  untouched. `I` excludes `(1,...,1)` — that is the leading term, fixed weight 1.
- **Nothing new is assembled.** Each term is still a product over axes, so
  separability survives: powers are elementwise on the interpolated values, one
  `QuadratureAssembly` per monom as before, and `C` never reaches one.
- Exponent-set builders: `uniform_exponents(d, n)` = `{(p,..,p) : p in 2..n}`;
  `total_degree_exponents(d, p)` = `{lambda_j >= 1, sum <= p}`. Zero exponents
  are **forbidden** — `w^0 = 1` cannot satisfy an axis' essential BC — which
  makes the total-degree set empty unless `p > d` (raises, naming the bound).
  For the 5-parametric beam that means `p >= 6`.
- Coefficients are per-mode `nn.Parameter` rows controlled by `requires_grad`
  alone — **no `active` flag**. Justified: `C` is zero-init and
  `d/dw [C prod w^lambda] ∝ C`, so a frozen `C = 0` perturbs neither the energy
  nor the monom gradients. Verified, not assumed: `test_zero_coefficients_do_not
  _perturb_monom_gradients` compares monom gradients against `CPPGD` on the real
  (layout) path and finds them equal.
- Consequence: the two-stage protocol needs no new machinery —
  `add_mode()` (unchanged, never releases `C`) → train as pure CP →
  `unfreeze_mode_coefficients(m)` → train the correction. Covered end to end.
- `polynomial_directory()` is the new seam for a non-linear energy: a flat list
  of `(mode, exponents, coefficient_or_None)`, read alongside the unchanged
  `directory()`. **The consuming double loop is quadratic in the term count**
  `n_modes*(1+|I|)` — 5 modes with `|I|=5` is 900 pairs where `CPPGD` has 25.
- **Gauge degeneracy — exact, verified, not corrected.** The energy is invariant
  under the *per-axis* rescaling

      w_ij -> s_j w_ij,  C_{i lambda} -> C_{i lambda} prod_j s_j^(-lambda_j),
      subject to prod_j s_j = 1

  Verified numerically: the energy is bit-identical along it (rel. diff `0.00e+00`
  for both exponent sets), while dropping the `prod_j s_j = 1` constraint does
  change it (3193.69 -> 7956.49). So the orbit has **d-1** dimensions per mode,
  not d — pinning the leading term's coefficient at 1 already fixes one. The
  Hessian is correspondingly singular in `d-1` directions per mode, which is the
  practical concern for a curvature-based optimiser.
- **Not** measured: that this drifts in practice. `test_report_gauge_drift`
  prints `||w||`/`max|C|` over 20 SGD steps, but that run does not demonstrate
  drift — with uniform exponents `lambda_x = lambda_E`, so `C` is *invariant*
  along the flat direction and its growth there is ordinary training from 0.
  Treat the degeneracy as proven and its practical cost as unmeasured.
- Vector-valued axes are **rejected** (`dim > 1` raises): `w^lambda` for a vector
  monom has no defined meaning here. Tightens `CPPGD`'s "at most one".
- `evaluate`/`assemble` stay detached (inherited `PointWiseInterpolator`
  behaviour), so gradient tests must go through the `FieldLayout`, and
  `backward(retain_graph=True)` is required as in `PGDTrainer._closure`.

## 2026-07-28 — amplitude/shape seed for new modes (5-parametric beam)

**State:** `docs/examples/1d_5-parametric_beam_PGD/` only —
`1d_5-parametric_beam_deflection_PGD.py` (seeding + `RunConfig`/`build_problem`
knob) and `sweep.py` (`show_ledger` seed column). No `src/` change: the
mechanism (`Axis.init_values_rest`) already exists. Not yet run/measured.

- New modes (`m >= 1`) now seed as an amplitude/shape split: parametric factors
  at unit shape (1.0), the product's small initial amplitude carried by the
  space factor (`seed_amplitude`, default 0.05). Mode 0 keeps the 0.5 seed.
- Replaces the uniform 0.5-on-every-factor seed, which spiked the loss on
  enrichment. Tutor's "seed the space mode at 0" collapses instead: an all-zero
  factor is a stationary point (`CPPGD.add_mode`); small-but-nonzero avoids both.
- `RunConfig.seed_amplitude` added → new `config_id`s, so every existing ledger
  row is stale (to be rerun). The knob rides into the ledger via `asdict`; old
  rows without it fall back in `show_ledger`.
- Untested claim: whether 0.05 is the right amplitude — a `seed_amplitude` sweep
  is the intended next step; watch `max_correlation`, not just the energy.

## 2026-07-28 — `Axis.init_values_rest`: distinct seed for modes after the first

**State:** `src/neurom/decompositions/pgd.py` changed only (`Axis` gains one
optional field, `CPPGD.__post_init__` reads it). No test or caller passes it
yet, so behaviour is unchanged unless an axis opts in.

- `init_values_rest`, defaults `None` (reuses `init_values` for every mode —
  the old, uniform behaviour). When set, mode 0 seeds from `init_values` and
  modes `m >= 1` seed from `init_values_rest` instead.
- Motivated by [pgd-greedy-degenerates-with-short-stages]: initially-active
  modes sharing one seed see the same gradient and can start correlated.
  Giving later modes a different starting point is untested here — no run
  measured yet with it actually set to a different value.

## 2026-07-28 — LHS reference parameter sampling for the 5-parametric beam

**State:** `docs/examples/1d_5-parametric_beam_PGD/reference/reference_fem_solution.py`
updated; `reference_solution.pt` regenerated (302 solutions: 300 LHS + 2
extreme points). `scipy` added as a project dependency. All 7 reference tests
pass.

- `sample_parameters` now draws points via `scipy.stats.qmc.LatinHypercube`
  (per-axis stratified, shuffled independently per axis) instead of plain
  uniform draws; it replaces `grid_parameters` (removed) as the source of the
  reference set's non-extreme points.
- `generate()` uses `N_LHS = 300` points in the 4D (E1, E2, alpha, n) box
  instead of `N_GRID = 5`'s full tensor grid (625 solves) — about half the
  solves, with much more even per-axis coverage than plain random draws.
  Labels for these points changed from `"grid-i"` to `"lhs-i"`; metadata
  keys `n_grid`/`grid_shape` replaced by `sampling`/`n_lhs`/`seed`. Checked:
  nothing downstream reads the old label prefix or metadata keys.
- Regenerated bundle: constant-modulus sanity check 6.27e-06 relative L2
  error (was 6.271e-06 pre-change, i.e. unaffected — that check doesn't
  depend on the parameter-sampling strategy). Not yet compared against a
  PGD decomposition to see whether `overall` error moves versus the old
  grid-based bundle.
- Not yet regenerated or measured against the old grid-based bundle.

## 2026-07-28 — Hyperparameter sweep + config ledger for the 5-parametric beam

**State:** `docs/examples/1d_5-parametric_beam_PGD/sweep.py` added;
`1d_5-parametric_beam_deflection_PGD.py` refactored to route training through a
`RunConfig`; `tests/test_sweep.py` added (8 tests). Both scripts run standalone.
No measured sweep results yet — this is the harness only (assembly + tests).

- **`RunConfig`** carries every training knob explicit and flat: strategy,
  `stage_tol/window/max_iter/min_iter/stage_floor` (RelativeChange),
  `enrichment_tol/enrichment_floor` (RelativeGain), `lr`, `n_modes_max`,
  `n_nodes`. `build_criteria`/`build_optimizer_factory` turn it into live
  objects; `main()` uses the same builders, so CLI and sweep cannot drift.
- **Identity is a content hash** (`config_id`, sha256 of the sorted fields):
  same knobs → same id (idempotent), any knob change → new id. `name` is a
  label only. Ledger rows and checkpoints (`pgd5_sweep_<id>.pt`) key off it.
- **Ledger** is JSONL, path is a `run_sweep` argument (a bigger, non-parameter
  change targets a fresh file). One row per config: full `config` + `result`
  (overall/worst/per-point L2, final energy, n_modes/stages, stop reason,
  total iters, and per-stage `amplitude`/`max_correlation`). `upsert_row`
  replaces by id, so editing `CONFIGS` never loses past results.
- **Committed to git** (the experiment record); `pgd5_sweep_*.pt` is ignored.

## 2026-07-27 — Dense reference grid + vectorised `mesh.elements_at`

**State:** `src/neurom/meshes/mesh.py` `elements_at` rewritten (vectorised);
example's `relative_errors` and `reference_fem_solution.py` updated. The
5-parametric example's report runs standalone again. Reference bundle
regenerated to a 627-point grid.

- **`elements_at` was O(points × elements) Python double loop** — stalled at the
  ~63k query points the report now issues (627 params × 101 space). Replaced by
  a vectorised interval test: `left/right` per element, `(P, E)` inside-mask,
  `argmax` for the first (lowest-id) match — same semantics as the old `break`.
  **1D only, as before** (the interval test assumes a segment mesh); a 2D mesh
  still needs a proper point-in-cell locator. Library-wide speedup: every
  interpolation goes through this.
- **`relative_errors` looped `evaluate()` once per reference point** — now one
  `(K·P, 5)` call (`evaluate` was already vectorised over rows).
- **Reference set is now a full tensor grid** — `N_GRID = 5` inclusive points
  per axis (`grid_parameters`) → 625 grid points + 2 `EXTREME_POINTS`, so
  `overall` is a genuine global L2 over the box, not 10 random draws.

| measure (simultaneous ckpt) | value |
|---|---|
| `relative_errors` over 627 pts | 0.24 s (was >120 s / stalled) |
| overall relative L2 (all axes) | 0.0746 |
| per-point min / median / max | 0.0032 / 0.0281 / 0.3888 |
| constant-modulus FEM check | 6.27e-06 |

## 2026-07-24 — Repo hygiene: example tests moved out of the library suite, CLAUDE.md rewritten, notes split out

**State:** `CLAUDE.md` rewritten (814 → 379 words) and now carries the changelog
format rules. Example tests live in
`docs/examples/1d_5-parametric_beam_PGD/tests/`, registered in `pyproject.toml`'s
`testpaths`. Long-form notes in `docs/notes/`. No `src/` change.

- **Moved** `test_1d_5_parametric_beam_energy.py` and `test_greedy_trainer.py`
  from `tests/integration/` into the example's own `tests/` directory — they
  load an example script by path and are not library tests. Path anchors now
  resolve from the example dir (`parents[1]`) rather than the repo root.
  `testpaths` extended so a bare `pytest` still collects them.
- **`CLAUDE.md`**: prose → numbered rules, layout as a table, references moved
  to `docs/notes/references.md`. Added two rules that previously lived only in
  session context: stage explicit paths (never `git add -A`), and the test
  placement convention above.
- **Changelog format is now specified and capped** — ≤12 bullets, ≤40 lines,
  a mandatory **State** line first, numbers in tables, overflow to
  `docs/notes/<date>-<topic>.md`. Prompted by this log becoming too long to
  read. The five entries from 2026-07-22..23 were retrofitted; their full
  reasoning, ablations and rejected approaches moved to `docs/notes/` verbatim.
- **New rule: measurements vs. thresholds.** `assert error < 0.15` is a bound,
  not a result. Both sides of any better/worse claim must be measured and
  quoted. This was written after exactly that error was made and caught below.

## 2026-07-24 — Simultaneous training: keep every mode trainable

**State:** `src/neurom/training/simultaneous.py` and `diagnostics.py` added;
`SimultaneousTrainer` selectable from the example
(`python 1d_5-parametric_beam_deflection_PGD.py simultaneous`).

- **Added** `SimultaneousTrainer`. Same enrichment schedule as `GreedyTrainer`,
  but `prepare_stage` never calls `freeze_all`: stage *m* solves the full
  rank-*m+1* problem, not just the residual. The explicit unfreeze loop is not
  redundant with `add_mode` (which unfreezes only what it adds), so a
  decomposition handed over frozen by a greedy run is picked up correctly — the
  two strategies compose.
- **Refactor**: duplication diagnostics moved out of `GreedyTrainer` into
  `src/neurom/training/diagnostics.py`. Greedy still reports the newest mode
  against its predecessors — nothing else can move. Simultaneous reports the
  worst **pairwise** correlation: with nothing frozen, two *earlier* modes can
  collapse onto each other long after either was added, which a
  newest-mode-only measure cannot see.

**Result** (default meshes, 300-iteration stages, same criteria, scored on
`reference_solution.pt`):

| | rank | overall | worst point | energy |
|---|---|---|---|---|
| simultaneous | 8 | **2.24e-2** | 3.52e-2 | -2.0522e11 |
| greedy | 8 | 4.17e-2 | 1.15e-1 | -2.0451e11 |
| simultaneous, own stop | 7 | **2.66e-2** | 5.69e-2 | — |
| greedy, own stop | 8 | 3.11e-2 | 6.13e-2 | — |

- **Negative result — stage length is not shareable between the two.** At
  greedy's `min_iter=120`, a simultaneous stage spends its budget re-fitting the
  old modes and the new one never takes off (amplitude 3e1 vs mode 0's 3e5; gain
  1.7e7 vs greedy's 5.6e9), so `RelativeGain` calls the run converged after
  *two* stages for a poor 1.19e-1. Hence `STAGE_MIN_ITER = {GreedyTrainer: 120,
  SimultaneousTrainer: 300}`. **Do not retry a shared budget.**
- **Also observed, unexplained**: even at 300 iterations roughly every other
  simultaneous stage adds a mode that stays at seed size (amplitude ~1), its
  gain coming from re-fitting existing modes. And greedy at rank 8 scores
  *worse* than at rank 6 (4.17e-2 vs 3.21e-2) despite a lower energy — **lower
  energy is not monotonically better accuracy** at these sampled points.

## 2026-07-24 — Train once, plot many times: checkpoints and a progress bar

**State:** `src/neurom/training/checkpoint.py` and `progress.py` added. The
example runs as
`python 1d_5-parametric_beam_deflection_PGD.py [greedy|simultaneous] [--retrain]`;
checkpoints (`pgd5_<strategy>.pt`) are gitignored.

- **Added** `save_checkpoint` / `load_checkpoint`. Saves `model.state_dict()`
  plus the `TrainingHistory` and a free-form metadata dict. The truncation needs
  no bookkeeping of its own: `QuadratureAssembly.active` is a registered buffer,
  so how many modes were active round-trips with the values. Geometry is
  deliberately *not* saved — `load_checkpoint` fills a freshly built model, and a
  mismatch raises from `load_state_dict` instead of silently plotting half a
  decomposition.
- **Added** `ProgressReporter` (no-op, the default, so tests and scripts stay
  silent) and `ProgressBar`, a stderr bar redrawn in place, one line per stage,
  finalised with the stage's stop reason. No tqdm dependency.
  `StageCriterion.budget()` is new and advisory: it is what lets the bar show a
  fraction; a criterion with no cap returns `None` and the bar degrades to a
  counter rather than lying.
- Without `--retrain` an existing checkpoint is loaded and only the figures are
  redrawn — iterating on a plot no longer costs a training run. The trained and
  reloaded paths share one `_report`, so a checkpoint reproduces exactly the
  output of the run that wrote it.

## 2026-07-24 — Two hard reference points instead of ten superposed curves; two error measures

**State:** `reference_solution.pt` regenerated (12 points); `plot_solution`
rewritten; `relative_errors` added.

- `reference_fem_solution.py` now appends two named `EXTREME_POINTS` to the ten
  random ones (12 total): `soft-stiff-left` (`E1=10, E2=100, alpha=3, n=5`) and
  `stiff-soft-right` (`E1=100, E2=20, alpha=6.5, n=4`). Sharp transitions, and
  deliberately *not* mirror images — opposite contrast direction, different
  ratio, transition on either side of mid-span. The bundle gained `labels` and
  `metadata["highlight"]`.
- `plot_solution` no longer draws ten near-identical parabolas. It draws the two
  highlighted points as two columns: `E(x)` on top (the nonlinearity being asked
  for) and `u(x)` below, reference vs PGD, with the pointwise error on a twin
  axis — a small global L2 can hide a local failure at the transition.
- **Error definition clarified**, and both are now reported by `relative_errors`.
  `per_point[label]` is **space-only**: the L2 ratio over the 101-point `x` grid
  with the parameters frozen. `overall` flattens the whole `(12, 101)` table into
  a single ratio of norms, i.e. space and parameter points jointly. Neither
  weights the parameter volume — the reference points are samples, not a
  quadrature. The old code reported only the mean/worst of the per-point
  numbers, which weights a barely-deflected stiff bar the same as a soft one.

**Result** (4-mode budget, greedy, default meshes):

| measure | value |
|---|---|
| overall | 5.5e-2 |
| `soft-stiff-left` | 1.20e-1 |
| `stiff-soft-right` | 3.05e-2 |
| worst random point | 1.38e-1 |

- The PGD **over-deflects on the soft side of a sharp interface** — that is the
  failure mode to beat with the non-linear decompositions.

## 2026-07-24 — FEM reference solutions for the 5-parametric bar, and plotting against them

**State:** `docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py`
added, writing `reference_solution.pt` (12 kB, committed). Example gained
`plot=True`. `ElasticEnergy` in `src/` gained an optional modulus.

- One plain `FEMModel` solve per parameter point — 400 nodes, `TwoPoints1D`,
  LBFGS — at 10 seeded-uniform `(E1, E2, alpha, n)` points, sampled on a fixed
  101-point `x` grid. Saving it is the point: every decomposition strategy is
  scored against *the same* numbers without re-solving. Regenerate with
  `python .../reference_fem_solution.py` (~20 s); the points are seeded, so they
  stay comparable across runs.
- **Library change**: `ElasticEnergy(field, modulus=None)` now accepts an
  optional modulus — either a field looked up in the layout (like
  `LoadPotential`'s load) or a callable evaluated at the quadrature points.
  Default `None` keeps the old unit-modulus behaviour; every existing test passes
  untouched. The reference passes the modulus as a *callable*, so `E(x)` is exact
  at the quadrature points instead of piecewise-linearly interpolated.
- **The reference runs in float64** (scoped `double_precision()` context, cast
  back to the caller's dtype on load). In float32 it stalled at ~1e-3 relative:
  halving both moduli, which must scale `u` by exactly 2, was off by 3e-3, and
  the constant-modulus check sat at 2.7e-4. In float64 those are 8e-5 and
  **6.3e-6** — the reference is no longer the limiting error.
- **Discarded approach**: a first version computed the reference analytically by
  quadrature (`(E u')' = f` → `u = int (C + f s)/E(s) ds`). It was correct (6e-5
  vs the constant-`E` parabola) but it is not what the surrogate is
  approximating; the direct FEM solve is the right yardstick and is reusable.
- **Added** plotting: `plot_convergence` (energy vs iteration with stage
  boundaries), `plot_solution`, `plot_modes` (each mode's per-axis factor,
  normalised — CP factors are only defined up to a per-axis scale, so coinciding
  curves are the degeneracy `max_correlation` reports). matplotlib is imported
  lazily inside the helpers.
- **Result**: a 3-mode greedy CP-PGD (default 30/20/20/15/15 nodes,
  `RelativeChange(tol=1e-3, window=20, max_iter=400, min_iter=120)`) reaches
  **4.2 % mean / 13.8 % worst** relative L2 over the 10 reference points. The
  worst point is the stiffest one (`E1=97, E2=74`), where the PGD
  under-deflects; the softer points are at 1–2 %.

## 2026-07-24 — GreedyTrainer in the 2-parametric example, checked against the exact solution

**State:** `docs/examples/1d_2-parametric_beam_PGD/1d_beam_deflection_PGD.py`
runs standalone; now structurally parallel to the 5-parametric example
(`make_axis` / `Problem` / `build_problem`). 157 tests passing. Notes:
[2026-07-24-greedy-trainer-2-parametric.md](docs/notes/2026-07-24-greedy-trainer-2-parametric.md).

- Three copy-pasted 150-iteration blocks replaced by one
  `GreedyTrainer(...).enrich()` with `RelativeChange(tol=1e-3, window=20,
  max_iter=600, min_iter=120)` + `RelativeGain(tol=1e-3)`.
- **The only check in this branch of the *answer* rather than the machinery.**
  This beam is exactly rank-1 (`u = 0.5 q (x-x_min)(x-x_max)/E`), so
  `test_greedy_trainer_recovers_the_analytical_beam` compares against the closed
  form on a dense 60x40 grid.

| Relative L2 error | Baseline (hand-rolled) | GreedyTrainer |
|---|---|---|
| mode 0 alone | 0.3269 | **0.1305** |
| greedy, rank 3 | **0.0954** | **0.1019** |
| after all-modes polish | **0.0153** | not performed |

- **The trainer is marginally worse, by ~7 % relative.** An earlier version of
  this entry claimed the opposite by comparing 0.1019 against
  `final_error_tol=15%` — a *tolerance*, not a measurement. Corrected.
- Stage 0 converges much *better* than baseline yet rank 3 ends slightly worse:
  on a rank-1 truth a stronger mode 0 leaves modes 1-2 fitting numerical
  residue. An all-modes polish stage is the natural fix.
- `ANALYTICAL_ERROR_TOL = 0.21`, ~2× the measurement. Untrained scores 1.0006 so
  the bound is real — but mode 0 alone scores 0.130, so this test exercises
  stage-0 convergence, not enrichment.
- Trained `max_correlation`: 0.000 / 0.242 / 0.161 — no duplication here, unlike
  the 5-parametric problem at short stages. Stopped at `"capacity"`.

## 2026-07-24 — GreedyTrainer wired into the 5-parametric example

**State:** the 5-parametric example runs standalone and now trains
(`main(train=True)`; `train=False` reproduces the old untrained behaviour). 18
tests. Plotting and a reference solution still absent at this point. Notes:
[2026-07-24-greedy-trainer-5-parametric.md](docs/notes/2026-07-24-greedy-trainer-5-parametric.md).

- `Problem` gained a `history` field (`None` unless trained). Per-stage
  diagnostics table printed by `main`.
- **`min_iter=120` keeps modes distinct on the full-size mesh**, confirming the
  tiny-mesh negative result's fix. `max_correlation` per stage: 0.000, 0.086,
  0.159, 0.114, 0.145, 0.198, **0.433**, 0.086 — all far below the 1.0 seen at
  `FixedIterations(15)`. Single run, not ablated across seeds; stage 6's 0.433 is
  worth watching.
- Energy fell monotonically `-1.905e11` → `-2.045e11` over 8 stages; gains
  `5.6e9` → `9.1e7`; `RelativeGain` stopped enrichment at stage 8
  (`"converged"`, capacity `n_modes_max=10` not reached).
- All 8 stages converged in exactly 120 iterations — the first eligible check.
- **`min_iter` ablated:** at `min_iter=1`, six of eight stages falsely fire
  `"converged"` at iteration 21 (relative change dips to 3.8e-5, climbs back to
  0.023 by n=50 as Adam escapes). **`tol` is not decorative either:** extending
  stages to n=300/400 shows the change shrinking to 1e-6, so 120 is genuine
  convergence detection, not a disguised `FixedIterations(120)`.

## 2026-07-23 — PGDTrainer: a base class for interchangeable PGD strategies

**State:** new module `src/neurom/training/` (`base.py`, `greedy.py`,
`criteria.py`, `history.py`). 34 unit tests in `tests/unit/training/`, against a
synthetic loss — not real physics. Notes:
[2026-07-23-pgd-trainer-base.md](docs/notes/2026-07-23-pgd-trainer-base.md).

- `PGDTrainer` (ABC) owns `enrich()` → `stage()` → `step()`. Criteria are
  **injected**, not overridden, so strategy and stopping rule stay independent;
  both return a *reason string*, so history records why a run ended.
- **The base deliberately knows nothing about modes** — never calls
  `add_mode`/`mode_parameters`, never assumes stage index == mode index. The
  trainable set is read off `requires_grad` alone; all decomposition manipulation
  lives in `prepare_stage`. This is what should let the same loop drive the
  non-linear decompositions, whose terms are not all modes.
- `step()` is closure-based (LBFGS works unbranched) and therefore returns the
  loss *before* that iteration's update — hence `StageRecord.energy` prefers a
  separately recorded `final_energy`. Using `losses[-1]` would put a one-update
  error into every `gain` and every enrichment decision.
- `RelativeChange` needs `min_iter` and a denominator floor of 1.0: without the
  floor, the energy crossing zero (+2e5 to -1e8) blows the denominator up and the
  stage never stops.
- **Design claim partly confirmed:** writing a `SimultaneousTrainer` needed only
  `prepare_stage`/`should_add_stage`. An `AlternatingTrainer` does not fit.
- **Open at the time:** `CPPGD` has no per-monom freeze, so alternating
  directions must reach into `monoms[m][k].values_reduced` — `freeze_monom(m, k)`
  is the missing seam. `on_stage_end` runs before `history.append`, so
  `record.gain` is NaN inside the hook. Divergence is detected but **not
  latched** — a second `enrich()` resumes the poisoned state.

## 2026-07-23 — GreedyTrainer, and two base.py bugs only real physics exposed

**State:** `src/neurom/training/greedy.py` added; 155 tests passing (146 + 9).
Notes:
[2026-07-23-greedy-trainer-and-base-bugs.md](docs/notes/2026-07-23-greedy-trainer-and-base-bugs.md).

- One stage = one mode: stage 0 trains the initially-active mode(s), later stages
  `freeze_all()` then `add_mode()`. `should_add_stage` checks capacity itself so
  `add_mode()`'s `RuntimeError` is never hit.
- `on_stage_end` records two per-stage diagnostics from raw nodal vectors:
  `amplitude` and `max_correlation`. Meant to **spot**, not quantify, a greedy
  step rediscovering an existing mode.
- **Two pre-existing `base.py` bugs, both invisible to the synthetic stub and
  both hit by every new test on first run:** `step()` needs `retain_graph=True`
  (the `QuadratureContext` subgraph is built once at construction and never
  rebuilt — *not* about `jacobian_field`, which was the first and wrong
  explanation); and `_record_final_energy`'s `torch.no_grad()` is fatal to any
  `jacobian_field` energy, fixed with `.detach()`.
- **Negative result — short stages make greedy enrichment degenerate.** At
  `FixedIterations(15)`, `max_correlation` reads 1.0 and stage energies come out
  exactly 1×/2×/3×: three copies of one mode.

| Iterations/stage | 5 | 15 | 40 | 80 | 200 |
|---|---|---|---|---|---|
| `max_correlation` | energies rise | 1.0 | 0.9999 | 0.0092 / 0.176 | 0.066 / 0.135 |

- ~80 iterations is where modes separate here. **Do not read a falling energy as
  successful enrichment — check `max_correlation`.**
- **Cause not established.** The seed-symmetry story was never ablated. One
  forced case: at `n_modes_ini=2`, stage 0 reports `max_correlation = 1.0` —
  initially-active modes share a seed and see the same gradient, with no freezing
  to break the symmetry.

## 2026-07-22 — 5-parametric beam with tanh-graded modulus

**State:** new example
`docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`,
runs standalone (assembly + energy only; training added 2026-07-24). 17 tests.
Nothing added under `src/`. Notes:
[2026-07-22-5-parametric-beam-energy.md](docs/notes/2026-07-22-5-parametric-beam-energy.md).

- Two-zone modulus `E = (E2-E1)/2 tanh(n(x-alpha)) + (E2+E1)/2`, giving five
  coordinates `(x, E1, E2, alpha, n)` and `u = sum_i X_i lambda_i mu_i A_i N_i`.
- **The energy is injected** into `build_problem(loss_fn, ...)`, so the same
  wiring will drive the non-linear PGD functionals unchanged. `physics.Term` is
  single-axis and does not fit a multi-axis separated energy; extending it is
  deliberately deferred.
- **`tanh(n(x-alpha))` is not separable** — it couples x, alpha and n, and is
  integrated by an exact 3-D tensor-product quadrature over those axes'
  quadrature points, rebuilt each call so r-adaptive meshes stay correct.
  Rejected: offline low-rank separation of `tanh` (extra error), Monte-Carlo
  (noisy gradients).
- **Verified against a brute-force 5-D quadrature that exploits no separability
  at all** — agreement to 1e-9 in float64, for 1 and 2 modes, both `MidPoint1D`
  and `TwoPoints1D`, and still after `add_mode()`. Two deliberate bug injections
  confirmed the tests bite. The printed `energy = 6.602113e+07` is reproduced by
  hand to 5 significant figures.
- `build_problem` takes an injectable `quad` so the `N_q > 1` path is testable —
  one quadrature point per element previously hid a broadcasting bug in the
  2-parameter ancestor.
- Ranges: `x in [0,10]` (30 nodes), `E1, E2 in [10,100]` (20), `alpha in [2,8]`
  (15), `n in [0.5,5]` (15). **`n` never reaches 0**, so the near-uniform regime
  is unsampled. Mode budget 10, on an untested assumption.
- **Known scaling limit, the first thing that will break:** the tanh grid is
  built inside the autograd graph and rebuilt every iteration — 5 684 entries at
  the defaults, but ~3M at ~300 space elements. The mode-pair loop also ignores
  `(i,j)`/`(j,i)` symmetry, and the grid hard-codes mode 0's quadrature points,
  so per-mode meshes would corrupt the energy **silently**.
