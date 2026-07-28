# Hyperparameter sweep — usage guide

How to run hyperparameter sweeps of the 5-parametric beam PGD and keep a
traceable record of which config produced which result.

## 1. Where you enter configs

Open [sweep.py](sweep.py) and edit the `_configs()` function — that list *is*
the run queue:

```python
def _configs():
    ex = _example()
    return [
        ex.RunConfig(name="baseline-greedy"),
        ex.RunConfig(name="tight-tol", stage_tol=1e-6, min_iter=200),
        ex.RunConfig(name="sim-longer", strategy="simultaneous", min_iter=300, lr=0.05),
    ]
```

Every knob is a keyword argument with a default, so you only write the ones
you're changing. Full list:

| knob | default | what it tunes |
|---|---|---|
| `name` | *(required)* | human label only — not identity |
| `strategy` | `"greedy"` | `"greedy"` or `"simultaneous"` |
| `stage_tol` | `1e-5` | RelativeChange plateau threshold |
| `window` | `20` | iterations the plateau is measured over |
| `max_iter` | `600` | hard cap per stage |
| `min_iter` | `120` | iterations before plateau test applies |
| `stage_floor` | `1.0` | RelativeChange denominator floor |
| `enrichment_tol` | `1e-5` | RelativeGain stop-enriching threshold |
| `enrichment_floor` | `1.0` | RelativeGain denominator floor |
| `lr` | `0.1` | Adam learning rate |
| `n_modes_max` | `10` | mode budget |
| `n_nodes` | `None` (→ `DEFAULT_N_NODES`) | per-axis mesh, e.g. `{"space":30,"E1":20,...}` |

## 2. Run the sweep

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam_PGD/sweep.py
```

It trains each config that isn't already recorded, saves a checkpoint per
config, appends a row to `sweep_results.jsonl`, then prints a comparison table
sorted best-first.

To force re-training of everything (overwriting rows + checkpoints):

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam_PGD/sweep.py --retrain
```

**Prerequisite:** the FEM reference must exist (errors are scored against it).
Generate it once:

```bash
.venv/bin/python docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py
```

## 3. The key behavior — nothing is ever lost

Results are keyed by a **content hash** of the config, not by `name`:

- **Re-run the same config** → same hash → it's cached, skipped (seconds, not
  minutes). The printout says `cached  : <name>`.
- **Change any knob** → new hash → a new row is written *beside* the old one.
  The old result stays.
- **Remove a config from the list** → its ledger row is untouched. The queue is
  just "what to compute now"; the ledger is the permanent record.

So you can freely add/edit/delete entries in `_configs()` across sessions and
the ledger accumulates every experiment.

## 4. Reading results back

The comparison table prints automatically at the end of a run. To inspect
anytime without retraining, from a Python shell:

```python
import importlib.util, pathlib
p = pathlib.Path("docs/examples/1d_5-parametric_beam_PGD/sweep.py")
sw = importlib.util.spec_from_file_location("sw", p)
mod = importlib.util.module_from_spec(sw); sw.loader.exec_module(mod)

mod.show_ledger("docs/examples/1d_5-parametric_beam_PGD/sweep_results.jsonl")   # the table
rows = mod.load_ledger("docs/examples/1d_5-parametric_beam_PGD/sweep_results.jsonl")  # raw dicts
```

Each row is fully self-describing — its complete `config` plus a `result` block
with overall/worst/per-point L2 errors, final energy, mode/stage counts, stop
reason, total iterations, and per-stage `amplitude` + `max_correlation` (the
degeneracy tell). You never need to reopen a checkpoint to see what a config did.

## 5. A separate ledger for a bigger change

The ledger path is an argument. If you later change something structural (not
just a hyperparameter) and want a clean file, call the runner directly instead
of via `__main__`:

```python
mod.run_sweep(configs, ledger="sweep_results_v2.jsonl")
```

## Two notes

- `sweep_results.jsonl` is **tracked in git** (your experiment record travels
  with the branch); the `pgd5_sweep_*.pt` checkpoints are ignored.
- A zero space seed (`init_values["space"] = 0.0`) makes the deflection start
  identically zero, which zeroes the gradient w.r.t. the parametric monoms —
  the sweep won't train cleanly from there. Use a nonzero space seed before a
  real sweep.
