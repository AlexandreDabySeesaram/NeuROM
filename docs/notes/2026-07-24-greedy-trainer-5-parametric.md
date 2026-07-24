# 2026-07-24 — GreedyTrainer wired into the 5-parametric example

Long-form notes behind the CHANGELOG entry of the same name.

## The change

`docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
`main(verbose=True, train=True)` now trains after the untrained energy is
printed, via `GreedyTrainer` with
`RelativeChange(tol=1e-3, window=20, max_iter=600, min_iter=120)` and
`RelativeGain(tol=1e-3)`.

`Problem` gained a `history` field (`None` unless trained). `train=False`
reproduces the old, untrained behaviour so the pre-existing
energy-matches-printed-value test keeps its exact assertions.

## Verification

`test_1d_5_parametric_beam_energy.py`, +1 test, 18 total. It asserts that greedy
enrichment gets past stage 0 (`len(stages) > 1` **and**
`n_modes_truncated > 1`), that every later stage's energy is at or below stage
0's, that the concatenated losses stay finite, and that the table is printed.

**Note on a tautology:** `n_modes_truncated == len(history.stages)` is kept as a
consistency check but carries no information on its own — it is an architectural
tautology of `GreedyTrainer` at `n_modes_ini=1`, true whether or not a mode is
ever added. That is why the two separate `> 1` assertions exist; without them a
regression that disabled `add_mode` entirely would still satisfy every other
assertion, since stage 0 alone drops the energy from ~6.6e7 to ~-1.9e11 and
swamps what later stages do.

## Result — `min_iter=120` keeps modes distinct on the real (non-tiny) problem

At least in this one run. All 8 stages before `RelativeGain` stopped the run
converged in **exactly 120 iterations** — the criterion's first eligible check,
at `min_iter`, already cleared `tol` every time, consistent with the "Adam
sticky early phase" this criterion was designed around.

`max_correlation` per stage:

| Stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `max_correlation` | 0.000 | 0.086 | 0.159 | 0.114 | 0.145 | 0.198 | **0.433** | 0.086 |

Well below the 1.0 seen at `FixedIterations(15)` in the tiny-mesh work,
confirming that negative result and its fix **on the full-size mesh**, not just
the tiny test mesh.

Energy fell monotonically from `-1.905e11` to `-2.045e11` across the 8 stages;
gains shrank from `5.6e9` to `9.1e7`, and `RelativeGain(tol=1e-3)` then stopped
enrichment at stage 8 with reason `"converged"` (capacity, `n_modes_max=10`, was
not reached).

**Single run, not ablated across seeds.** Stage 6's jump to
`max_correlation=0.433` is the largest value seen and worth watching if this is
rerun.

## `min_iter` is load-bearing, and this was ablated

Replaying the same run with `min_iter=1`: stages 1, 3, 4, 5, 6 and 7 all falsely
fire `"converged"` at iteration 21.

Tracing stage 1: the relative change dips to `3.8e-5` at n=21 (a false plateau),
climbs back to `0.023` by n=50 as Adam escapes and the loss dives from
`-1.9046e11` to `-1.9593e11`, then decays back under `tol` around n=80. That is
the "Adam sticky early phase" the criterion was designed around, reproduced
concretely.

## `tol` is not decorative either

Extending stage 1 to n=300 and stage 6 to n=400 shows the relative change
shrinking to 1e-6–1e-7, and stage 6's `max_correlation` drifting only
0.433 -> 0.423. So stopping at 120 is **genuine convergence detection** rather
than a disguised `FixedIterations(120)`.

## Suite

156 passed (155 baseline + 1). Full report: `.superpowers/sdd/task-4-report.md`.
