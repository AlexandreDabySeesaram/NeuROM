# 2026-07-24 — GreedyTrainer in the 2-parametric example, validated against the analytical solution

Long-form notes behind the CHANGELOG entry of the same name.

## The refactor

`docs/examples/1d_2-parametric_beam_PGD/1d_beam_deflection_PGD.py`: the three
copy-pasted 150-iteration blocks (`freeze_mode` / `add_mode` /
`add_mode_to_optimizer`, one persistent `Adam` + closure reused across all
three) are gone, replaced by a single `GreedyTrainer(...).enrich()` call with
`DEFAULT_STAGE_CRITERION` (`RelativeChange(tol=1e-3, window=20, max_iter=600,
min_iter=120)`) and `DEFAULT_ENRICHMENT_CRITERION` (`RelativeGain(tol=1e-3)`).

Setup moved into `build_problem(loss_fn, *, n_modes_max=3, n_modes_ini=1,
n_nodes=None, quad=None) -> Problem`, mirroring the 5-parametric example's
`make_axis` / `Problem` / `build_problem` shape exactly — this example is now
structurally parallel to it.

`Problem` carries `x_min`, `x_max`, `E_min`, `E_max`, `load_value` alongside the
usual `model`/`pgd`/`field_layout`/`domain`/`axes`/`history`, since this is the
one problem whose exact analytical solution can be checked against those
scalars.

## This is the only task in the plan that checks the answer, not the machinery

The 2-parametric beam is exactly rank-1:

    u = 0.5 q (x - x_min) (x - x_max) / E

`test_greedy_trainer_recovers_the_analytical_beam` trains via `GreedyTrainer`
and asserts the trained model's relative L2 error against that closed form
(dense 60x40 (x, E) grid) is below `ANALYTICAL_ERROR_TOL`.

**Measured relative L2 error: 0.1019** (10.2%), identical across
`manual_seed(0)` and `manual_seed(42)` — this pipeline has no randomness beyond
the deterministic `0.5*ones` monom seed, so the run is fully reproducible, not a
lucky draw. `ANALYTICAL_ERROR_TOL` set to `0.21`, roughly double the
measurement, rounded up.

## Correction: tolerances are not measurements

An earlier version of the changelog entry claimed `GreedyTrainer` beat the
hand-rolled loop "without a regression", citing `final_error_tol` (15%) and
`strict_error_tol` (3%) in `tests/integration/test_1d_beam_deflection_PGD.py`.

**Those are *tolerances*, not *measurements*.** Comparing `GreedyTrainer`'s
0.1019 against them and calling it "better" was wrong.

Rerunning that baseline test with `-s` gives the actual numbers:
`rel. error per mode : [0.3269, 0.1807, 0.09536]`, `error after polish :
0.01527`.

| | Baseline (hand-rolled) | GreedyTrainer |
|---|---|---|
| mode 0 alone | 0.3269 | **0.1305** |
| after 2 modes | 0.1807 | — |
| greedy regime (rank 3) | **0.0954** | **0.1019** |
| after all-modes polish | **0.0153** | not performed |

So the trainer is **marginally worse in the greedy regime, by about 7%
relative**.

**What the comparison is against.** Not the old example's fixed-150-iteration
loop — no measurement here comes from that. It is against the *baseline test*,
which runs its own hand-calibrated plateau rule (`epsilon=2e-2`,
`plateau_window=20`, `min_epochs_per_mode=120`, `max_epochs_per_mode=600`). That
rule and `RelativeChange(tol=1e-3, window=20, min_iter=120, max_iter=600)` are
nearly the same criterion; they differ mainly in `tol` (1e-3 vs 2e-2), which
makes the two numbers a fair like-for-like comparison rather than a confound.

The polished baseline is 0.0153 — an extra all-modes joint optimization stage
`GreedyTrainer` does not perform, so the trainer does not yet close that gap
either.

## More interesting than the 7%: stage 0 converges much better

Mode-0-alone error **0.1305** vs. the baseline's **0.3269**, yet the final
rank-3 result is still slightly worse.

On a truth that is exactly rank-1, a better mode 0 leaves less real signal in
the residual for modes 1-2 to fit — they are left fitting numerical residue
either way, so a stronger mode 0 does not guarantee a stronger rank-3 sum.

A future `simultaneous`/`greedy+update` strategy — with an all-modes polish
stage — is the natural place to close both gaps.

## What that test does and does not discriminate

Measured rather than assumed:

- an untrained model scores `1.0006`;
- an undertrained one (<=50 iterations) scores `~1.0000`;

so `0.21` is a real bound, not a vacuous one.

**But mode 0 alone scores `0.130`, also under the bound.** The truth being
exactly rank-1, mode 0 is most of the answer. So this test mainly exercises
stage-0 convergence; the enrichment machinery is pinned separately by
`test_max_correlation_is_one_for_a_deliberately_duplicated_mode`. The
answer-check and the enrichment-check are not the same check here.

Relatedly, `torch.manual_seed(...)` pins nothing at present: nothing under
`src/neurom/` draws random numbers, and monoms start from a deterministic
`0.5*ones`.

## Trained diagnostics

| Stage | `max_correlation` | amplitude | gain |
|---|---|---|---|
| 0 | 0.000 (nothing to correlate against) | 7.04e3 | — |
| 1 | 0.242 | 1.34e3 | 2.46e5 |
| 2 | 0.161 | 4.17e2 | 3.98e3 |

Both well below 1.0 — modes 1 and 2 do *not* duplicate mode 0 here, unlike the
5-parametric problem at short stage lengths.

Training stopped at `"capacity"` (`n_modes_max=3` reached) rather than
`RelativeGain` converging first, so a larger mode budget was not tried. With the
true rank being 1, modes 1 and 2 are pure numerical residue, and their shrinking
amplitudes and gains are consistent with that, not with degeneracy.

## Suite

157 passed (156 baseline + 1). Full report: `.superpowers/sdd/task-5-report.md`.
