# 2026-07-23 — PGDTrainer: a base class for interchangeable PGD training strategies

Long-form notes behind the CHANGELOG entry of the same name.

## The module

`src/neurom/training/`:

- **`base.py`** — `PGDTrainer` (ABC) owns the loop `enrich()` -> `stage()` ->
  `step()`: a run is a sequence of stages, each iterating `step()` until its
  stage criterion fires, until an enrichment criterion (or the strategy) says to
  stop adding stages.
- **`history.py`** — plain-data records. `StageRecord` (one stage's losses, stop
  reason, end-of-stage energy, gain, diagnostics) and `TrainingHistory` (the
  ordered list plus a run-level stop reason).
- **`criteria.py`** — two protocols, `StageCriterion` (watches a stream of
  per-iteration losses) and `EnrichmentCriterion` (watches completed stages),
  and four concrete criteria: `RelativeChange`, `FixedIterations`,
  `RelativeGain`, `MaxStages`.
- **`greedy.py`** — `GreedyTrainer`, the first concrete strategy. See
  `2026-07-23-greedy-trainer-and-base-bugs.md`.

## The load-bearing design choice: the base knows nothing about modes

It never calls `add_mode` or `mode_parameters`, and never assumes a stage index
is a mode index. The trainable set for each stage is read off
`[p for p in model.parameters() if p.requires_grad]`.

All decomposition manipulation — freezing, adding a mode, building the optimizer
— lives in the `prepare_stage` hook that subclasses implement.

This is what is meant to let the same loop drive the planned non-linear
decompositions, whose trainable terms are not all modes.

## Criteria are injected objects, not overridden methods

Overriding would need a subclass per (strategy × stopping rule) pair —
`GreedyWithRelTol`, `GreedyWithFixedIter`, `AlternatingWithRelTol`, ... —
whereas injecting `stage_criterion`/`enrichment_criterion` keeps the strategy
and the stopping rule independent.

Both return a **reason string** rather than a bool, so
`TrainingHistory`/`StageRecord` record *why* a run or stage ended
(`"converged"`, `"max_iter"`, `"n_iter"`, `"capacity"`, `"diverged"`), not just
that it did.

## `step()` is closure-based

`self.optimizer.step(closure)`, so LBFGS works with no branching.

Consequence: it returns the loss *before* that iteration's update — that is the
contract of `optimizer.step(closure)`. That is why `StageRecord.energy` prefers
a separately recorded `final_energy` (`_record_final_energy`, one extra forward
pass after the stage's criterion fires) over `losses[-1]`. Using the last
recorded loss as "the stage's energy" would put a one-update error into every
`gain` and every enrichment decision.

## `RelativeChange` needs `min_iter` and a denominator floor of 1.0

Both carried over unchanged from the criterion calibrated by hand in
`tests/integration/test_1d_beam_deflection_PGD.py`.

- **Without `min_iter`:** Adam's long sticky early phase on this problem (energy
  barely moves for ~100 iterations before it escapes and dives) reads as a
  plateau and stops mode 0 after ~20 iterations.
- **Without the floor:** the energy crossing zero (~+2e5 to ~-1e8) makes a
  purely relative denominator blow up near the crossing, and the stage never
  stops.

## Verification

34 unit tests in `tests/unit/training/` (`test_base.py`, `test_criteria.py`,
`test_history.py`), against a **synthetic model/loss, not real physics** —
which is precisely why the two bugs in the companion note went unseen here.

Covered: loop monotonicity; frozen parameters left bitwise unchanged; every stop
reason firing (`converged`, `max_iter`, `n_iter`, `capacity`, `diverged`); a
diverging stage stopping the whole run rather than poisoning later stages; and
`enrich()` being resumable (calling it again continues from
`len(history.stages)`).

## Independent confirmation from the final review's own experiment

Writing a `SimultaneousTrainer` against this API needed only `prepare_stage` and
`should_add_stage` overrides — confirming the "the base knows nothing about
modes" design claim for that family of strategies.

But an `AlternatingTrainer` needs seams that do not exist yet.

## Not yet done

- **`CPPGD` has no per-monom freeze.** It exposes
  `freeze_all`/`freeze_mode`/`unfreeze_mode` but nothing per-monom, so an
  alternating-directions strategy (sweeping one axis of one mode at a time) has
  to reach past the public API into `monoms[m][k].values_reduced` directly. A
  `freeze_monom(m, k)` / `unfreeze_monom(m, k)` pair on `CPPGD` is the missing
  seam.
- **`on_stage_end` runs *before* `history.append`**, so inside that hook
  `record.gain` is still NaN — it is only filled in by `TrainingHistory.append`
  afterwards. A strategy wanting the gain inside `on_stage_end` currently cannot
  get it there.
- **Divergence is detected but not latched.** `enrich()` breaks out of the loop
  on a diverged stage, but calling `enrich()` again resumes on the poisoned
  state and burns another mode; `history.stop_reason` is silently overwritten.
  `RelativeGain` cannot stop it either — `(previous - current) / denom < tol` is
  False under NaN, so a NaN history reads as "still gaining". Self-limiting in
  practice (the poisoned state re-diverges at once, so it is not data
  corruption), but `enrich()` should refuse to resume a history whose last stage
  diverged.
- **A trainer for plain `FEMModel`** — its `forward()` returns the loss
  directly, which the current `step()` does not accommodate.
