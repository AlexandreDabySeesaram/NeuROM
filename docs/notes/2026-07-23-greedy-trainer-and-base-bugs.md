# 2026-07-23 — GreedyTrainer, and two base.py bugs only real physics exposed

Long-form notes behind the CHANGELOG entry of the same name.

## The strategy

`src/neurom/training/greedy.py`: `GreedyTrainer(PGDTrainer)`, the first concrete
strategy. One stage = one mode:

- stage 0 trains the initially-active mode(s);
- every later stage calls `freeze_all()` then `add_mode()` before building a
  fresh optimizer.

`should_add_stage` checks `n_modes_truncated >= n_modes_max` itself (stop reason
`"capacity"`) so `CPPGD.add_mode()`'s `RuntimeError` at capacity is never hit.

`on_stage_end` records two diagnostics per stage from raw nodal vectors (cheap,
no forward pass):

- `amplitude` = `prod_k ||w_m^k||`
- `max_correlation` = largest normalised rank-1 inner product against earlier
  modes

These are meant to **spot**, not quantify, a greedy step rediscovering an
existing mode — every mode shares the same `Axis.init_values` seed. The
diagnostic uses raw nodal vectors, not the quadrature-weighted L2 inner
product.

## Verification

`docs/examples/1d_5-parametric_beam_PGD/tests/test_greedy_trainer.py`, 7 tests
at the time of writing, against the real 5-parametric beam on a tiny mesh
`{space:5, E1:4, E2:6, alpha:7, n:3}`, `n_modes_max=3`:

- one mode added per stage;
- stage 0 does not enrich;
- frozen modes bitwise unchanged by later stages;
- capacity stops the run before `add_mode()` can raise;
- the first mode's `max_correlation` is exactly 0.

The end-of-stage-energy-never-rises property (`FixedIterations(80)`, tolerance
`1e-9 * |previous.energy|`) **passed unmodified** — 80 iterations was enough for
each stage to work off the seam jump from the new mode's nonzero seed.

**Caveat:** at the shorter `FixedIterations(15)` most tests use, the diagnostics
being "in range" hides that they are pinned at a *degenerate* value, not a
healthy one. See the negative result below.

## Two pre-existing `base.py` bugs found and fixed

Both invisible to the synthetic stub loss used in the base class's own unit
tests, and both hit by every single new integration test on first run.

### 1. `step()`'s `loss.backward()` needed `retain_graph=True`

**Cause:** `QuadratureContext.__init__` builds `x_phys` **and** `xi_back` once,
at construction, from the `requires_grad` leaf `_xi_ref` (`_compute_quad_pos`).
`NeuROMModel.forward` only calls `interpolate_all` and never
`IntegrationDomain.update_contexts()`, so every iteration's forward reads
through that same already-built subgraph. A non-retaining `backward()` frees it
after the first iteration, so the second iteration's backward raises "Trying to
backward through the graph a second time".

**This is not about `jacobian_field`'s `create_graph=True`.** That was the first
explanation offered and it is wrong. An energy with no `jacobian_field`, no
`create_graph`, and no reference to `_xi_ref` fails identically on iteration 2,
because it still reads interpolated quantities built on that one shared
subgraph. Confirmed by reproducing standalone with a bare `Adam` + closure
(2 steps, no trainer).

Expect this to bite *any* strategy trained against *any* energy that reads
interpolated fields — not just `GreedyTrainer`, and not just
autograd-differentiated ones.

### 2. `_record_final_energy` wrapped its forward pass in `torch.no_grad()`

`no_grad()` strips `grad_fn` from every intermediate tensor regardless of the
underlying leaves — fatal for any energy using `jacobian_field` internally
(`torch.autograd.grad` then has nothing to differentiate).

Fixed by dropping `no_grad()` and using `.detach()` on the returned loss
instead, matching the pre-trainer 2-parameter example's own
`loss.detach().item()` pattern.

Not anticipated by the task brief; a latent defect in already-committed code
that would otherwise have broken end-of-stage energy tracking
(`StageRecord.energy`, `gain`, `RelativeGain`) for every future strategy trained
against real physics.

## Negative result — short stages make greedy enrichment degenerate

On the tiny-mesh 5-parametric problem the new `max_correlation` diagnostic
immediately earned its keep. At `FixedIterations(15)` (what most of the tests
use) it reads **1.0** for stages 1 and 2, amplitudes match to 4 decimals, and
the stage energies come out exactly 1x/2x/3x — three copies of the same mode.

| Iterations/stage | `max_correlation` per stage |
|---|---|
| 5 | energies **rise** |
| 15 | 0.0 / 1.0 / 1.0 |
| 40 | 0.0 / 0.999998 / 0.999996 |
| 80 | 0.0 / 0.0092 / 0.176 |
| 200 | 0.0 / 0.066 / 0.135 |

So ~80 iterations per stage is where the modes actually separate on this
problem. **Do not read a falling energy as successful enrichment** — check
`max_correlation`.

The 15-iteration row was reproduced independently; the other rows are from a
single run.

### Cause not established

The plausible story is that every mode carries the same `0.5*ones` seed and that
early on the linear load term dominates, so a fresh mode retraces mode 0's
trajectory — but **that was never ablated**, only the duplication itself was
measured. Whether a per-mode seeding strategy is needed is still open, and so is
the mechanism.

**One case where the mechanism is forced rather than merely plausible:** at
`n_modes_ini=2`, stage 0 reports `max_correlation = 1.0` with amplitudes
matching to 4 significant digits. Every initially-active mode shares the same
`Axis.init_values` seed and sees the same gradient, so joint training keeps them
parallel forever. There is no freezing between the two modes to even
hypothetically break the symmetry.

## Suite

155 passed (146 baseline + 9), no regressions. Full report:
`.superpowers/sdd/task-3-report.md`.
