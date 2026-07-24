## 2026-07-24 — GreedyTrainer wired into the 5-parametric example

- **Changed** `docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
  `main(verbose=True, train=True)` now trains after the untrained energy is
  printed, via `GreedyTrainer` with `RelativeChange(tol=1e-3, window=20,
  max_iter=600, min_iter=120)` and `RelativeGain(tol=1e-3)`. `Problem` gained a
  `history` field (`None` unless trained). `train=False` reproduces the old,
  untrained behaviour so the pre-existing energy-matches-printed-value test
  keeps its exact assertions.
- **Verified** (`tests/integration/test_1d_5_parametric_beam_energy.py`, +1
  test, 18 total): the trained run's concatenated losses strictly improve and
  stay finite, `n_modes_truncated == len(history.stages)`, and the table is
  printed.
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
- Full suite: 156 passed (155 baseline + 1). Full report:
  `.superpowers/sdd/task-4-report.md`.

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
