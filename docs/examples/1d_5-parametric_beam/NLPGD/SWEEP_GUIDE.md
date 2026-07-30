# Sweeping the non-linear PGD

How to drive `sweep.py` in this directory. Scope: the `joint` and `refine`
schedules against the `greedy` CP baseline. `staged` is out — it has no working
`coefficient_lr` on this problem, so a row for it measures the parameterisation
rather than the schedule
(`docs/notes/2026-07-29-nl-pgd-coefficient-scale.md`).

Everything below assumes the repo venv:

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam/NLPGD/sweep.py
```

## What the runner is

A queue of `RunConfig`s plus a persistent JSONL ledger. The contract that makes
it usable:

- A config's **identity is the hash of its contents**, not its `name`. Rename
  freely; change any knob and you get a new row *beside* the old one.
- A config already in the ledger is **skipped**. Re-running the queue after
  adding one entry trains one config, not all of them.
- Every trained config writes `param_sweep/nlpgd5_sweep_<id>.pt`, so any row can
  be reloaded later without retraining.

So the ledger accumulates. Deleting an entry from the queue does not delete its
result; that is the point.

## The normal loop

**1. Edit the queue.** `_configs()` at the bottom of `sweep.py`. It currently
holds the three-way comparison:

| name | strategy | what it is |
|---|---|---|
| `cp-baseline-greedy` | `greedy` | never releases a coefficient row — this *is* CP-PGD, the zero line |
| `nl-joint-uniform3` | `joint` | monoms + `C` together, one stage per mode |
| `nl-refine-uniform3` | `refine` | CP stage, then monoms + `C` together |

All three share every other knob, so the spread in `overall_error` is
attributable to the schedule and nothing else. Keep it that way — vary one knob
at a time, in a new entry.

### Adding the `simultaneous` baseline

Yes, and it is a legitimate second baseline: `SimultaneousTrainer` also never
calls `unfreeze_mode_coefficients`, so it too runs pure CP-PGD on a
`PolynomialNLPGD`. It is the *stronger* CP line — it re-fits every earlier mode
at each stage instead of freezing them, so at equal rank it should beat
`greedy`. Measuring the NL schedules against it is the harder test.

**It needs its own `min_iter`.** This is the one knob the strategies must not
share. A greedy stage spends its whole budget on one new mode; a simultaneous
stage has to re-fit every earlier mode as well, and at 120 iterations the new
mode never takes off — amplitude `3e1` against mode 0's `3e5`, gain `1.7e7`
against greedy's `5.6e9`, small enough that `RelativeGain` calls the run
converged after two stages. At 300 it takes off and overtakes `greedy`.

The example's `STAGE_MIN_ITER` encodes that (300 for `SimultaneousTrainer`), but
**the sweep does not consult it** — a `RunConfig` is the single source of every
knob, so `cfg.min_iter` wins. Spell it out:

```python
ex.RunConfig(
    name="cp-baseline-simultaneous", strategy="simultaneous",
    min_iter=300, stage_tol=1e-5, enrichment_tol=1e-5,
),
```

Leave it at 120 and you get a converged-after-two-stages run that looks like a
weak baseline and isn't one.

**2. Run it.**

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam/NLPGD/sweep.py
```

Prints per-config progress, then the ledger sorted best-first. Add `--retrain`
to force-retrain everything in the queue (overwrites those rows and checkpoints;
leaves rows not in the queue alone).

**3. Read the table.**

```
name                 config_id     overall       worst  modes   iters    seed  I               stop
```

- `overall` / `worst` — relative error against the FEM reference, mean and
  worst reference point. This is the number you are comparing.
- `iters` — total optimizer iterations. **Read it next to `overall`.** `refine`
  is a two-stage schedule with the same `min_iter`, so it spends roughly twice
  the iterations per mode that `joint` does. Equal rank is not equal budget.
- `I` — exponent set and bound, e.g. `uniform3`. Parenthesised (`(uniform3)`)
  for `greedy`/`simultaneous`, which build the set but never use it.
- `stop` — `capacity` means it ran out of modes, not out of gain. If every row
  says `capacity`, `n_modes_max` is the binding constraint and the comparison is
  about the budget, not the schedules.

## The knobs worth sweeping

Ranked by how much they matter here.

| knob | default | notes |
|---|---|---|
| `strategy` | `"joint"` | `joint`, `refine`, `greedy`, `simultaneous` |
| `exponent_set` | `"uniform"` | `"uniform"` or `"total_degree"` |
| `max_power` | `3` | for `uniform`, gives `max_power − 1` terms per mode; for `total_degree`, must be **> 5** (five axes, every exponent ≥ 1) or the set is empty |
| `coefficient_lr` | `1e-3` | separate Adam lr for the `C` rows. The default is the best of five values on a *tiny* mesh — a starting point, not a tuned value. Sweep it. |
| `lr` | `0.1` | the monoms' lr, unchanged from the CP example |
| `n_modes_max` | `5` | cost is quadratic in the number of *terms*; see below |
| `min_iter` | `120` | per **stage**, not per mode |
| `stage_tol`, `enrichment_tol` | `1e-5` | plateau and gain thresholds |
| `seed_amplitude` | `0.05` | must stay > 0; an all-zero factor is a stationary point |

### Cost

The energy is a double loop over *terms*, and a mode carries `1 + |I|` of them.
With `uniform3` (`|I| = 2`), 5 modes means 15 terms, so 225 pairs — measured at
~110 ms/iteration, versus ~6 ms at one mode. `total_degree` at `max_power=6`
gives 6 terms per mode, four times the pair count of `uniform3`; drop
`n_modes_max` to 3 when you enable it, or a single config runs for hours.

## Inspecting a row afterwards

Any row reloads from its checkpoint in seconds. From a Python shell in this
directory:

```bash
.venv/bin/python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('sw', 'docs/examples/1d_5-parametric_beam/NLPGD/sweep.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.show_ledger('docs/examples/1d_5-parametric_beam/NLPGD/sweep_results.jsonl')"
```

Then, with `sw` imported:

- `sw.show_run(config_id)` → the **full training log**, see below.
- `sw.load_run(config_id)` → the trained `Problem`, with `.pgd` and `.history`.
- `sw.plot_losses(config_id)` → energy vs. iteration, into `plots/`.
- `sw.plot_extremes(config_id)` → PGD vs. reference at the two hardest points.
- `sw.delete_row(path, config_id)` → prune a stale row without hand-editing the
  JSONL.

`show_results.py` in this directory is the scratch driver for all of these —
uncomment a block, paste in ids, run it.

## Reading the training log after the fact

The `TrainingHistory` rides along in the checkpoint, so the log is not lost when
the run ends. `sw.show_run(config_id)` reprints it — the same table the run
itself printed, from the checkpoint, no retraining:

```
stage   kind  iters       stop         energy         gain    amplitude  max corr          |C|
    0  joint    600   max_iter  -1.234567e+11   5.6000e+09   3.0100e+05     0.000   4.1e-05
    1  joint    600   max_iter  -1.891234e+11   5.7000e+10   2.9000e+04     0.213   8.3e-06
...

polynomial coefficients C (one row per mode)
  exponents: [(2, 2, 2, 2, 2), (3, 3, 3, 3, 3)]
  mode  0:   4.0912e-05   1.1043e-09
```

Three columns are worth more than the ledger's summary:

- **`max corr`** — correlation of the new mode with the existing ones. Near
  `1.0` means the greedy step rediscovered an earlier mode and the "new" mode is
  a copy. This is the failure that short stages produce, and it does *not* show
  up as a bad energy.
- **`|C|`** — L1 norm of the stage's coefficient row. **All zeros means the run
  released nothing and is silently a CP run.** The expected reading for a
  `greedy`/`simultaneous` baseline row; a bug for `joint` or `refine`.
- **`kind`** — what the stage trained (`cp`, `corr`, `joint`). A baseline run
  has one stage per mode and every row reads `cp`.

The coefficient block underneath is the direct check on the scale argument in
the note: the `p=3` entry should sit several orders below the `p=2` one. If they
are comparable, or either is `O(1)`, the correction has taken over the mode
rather than correcting it.

Both `kind` and `coefficient_norm` are also stored per stage in the ledger's
`result.stages`, so a scripted comparison across rows does not need the
checkpoints — but only for rows written after this was added; older rows lack
the keys.

## Running one strategy without the sweep

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam/NLPGD/1d_5-parametric_beam_deflection_NLPGD.py joint --retrain
```

Writes `param_sweep/nlpgd5_<strategy>.pt` and the figures. Without `--retrain`
it loads the checkpoint and only redraws — changing a plot must not cost a
training run. Use this while iterating on one configuration; move to the sweep
when you want rows you can compare.

## Two traps

**The ledger is not comparable to `../PGD/sweep_results.jsonl`.** Configs here
carry `exponent_set` and `max_power`, so the content hashes differ even for
otherwise-identical runs. Compare the `overall` column across the two
directories, never the `config_id`.

**A CP baseline that silently stops being one.** `greedy` is the zero line
purely because the library's `GreedyTrainer` never calls
`unfreeze_mode_coefficients`. If that ever changes, the baseline rows quietly
stop being a baseline — which is why
`tests/test_polynomial_greedy_trainer.py::test_a_cp_baseline_trainer_leaves_every_coefficient_at_zero`
exists. If it fails, distrust every comparison in the ledger.
