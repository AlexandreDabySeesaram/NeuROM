# `neurom.training` — how training, stopping, checkpointing and reloading work

State of the module as of 2026-07-24 (branch `nl_pgd`). Everything below is
read off the source in `src/neurom/training/`; file/line references are
clickable.

---

## 1. What the module is for

A PGD run is **not** one optimisation. It is a sequence of *stages*, and
between stages the shape of the problem changes — a mode is activated, the
freeze state is rewritten, the energy gains a term. `neurom.training` owns
that outer structure:

```
enrich()                      # the run: loop over stages
  └── prepare_stage(i)        # STRATEGY: freeze state + fresh optimizer
  └── stage(i)                # the inner loop, until the stage criterion fires
        └── step()            # one optimizer iteration (closure-based)
  └── on_stage_end(record)    # STRATEGY: diagnostics
```

Everything variable is **injected**, not subclassed: the optimizer factory, the
stage criterion, the enrichment criterion, the progress reporter. Only the two
genuinely strategy-shaped decisions (`prepare_stage`, `should_add_stage`) are
abstract methods.

### Files

| File | Contents |
|---|---|
| [base.py](src/neurom/training/base.py) | `PGDTrainer` — the abstract template, the loop, `step`, divergence handling |
| [greedy.py](src/neurom/training/greedy.py) | `GreedyTrainer` — one mode per stage, predecessors frozen forever |
| [simultaneous.py](src/neurom/training/simultaneous.py) | `SimultaneousTrainer` — one mode per stage, **nothing** frozen |
| [criteria.py](src/neurom/training/criteria.py) | `RelativeChange`, `FixedIterations` (stage); `RelativeGain`, `MaxStages` (enrichment) |
| [history.py](src/neurom/training/history.py) | `StageRecord`, `TrainingHistory` — plain dataclasses, no torch |
| [diagnostics.py](src/neurom/training/diagnostics.py) | `amplitude`, `correlation`, `max_correlation`, `max_pairwise_correlation` |
| [checkpoint.py](src/neurom/training/checkpoint.py) | `save_checkpoint`, `load_checkpoint` |
| [progress.py](src/neurom/training/progress.py) | `ProgressReporter` (no-op default), `ProgressBar` (stderr, one line per stage) |

All public names are re-exported from `neurom.training`
([`__init__.py`](src/neurom/training/__init__.py)).

---

## 2. The contract with the model

A trainer needs a `model` that:

* is callable with **no arguments** in training mode and returns the filled
  `FieldLayout` — `NeuROMModel.forward` does `integration_domain.interpolate_all(...)`
  then returns the layout ([neurom_model.py:47](src/neurom/neurom_model.py:47));
* exposes `model.loss(output) -> 0-dim Tensor`;
* exposes `model.parameters()` with meaningful `requires_grad` flags;
* has `model.decomposition` (both concrete trainers read it in `__init__`).

The trainable set is never enumerated by the base class. It is *read off the
freeze state*:

```python
[p for p in model.parameters() if p.requires_grad]
```

([base.py:77](src/neurom/training/base.py:77)). This is the seam that will let
the same loop drive the non-linear decompositions: whatever `prepare_stage`
left unfrozen is what gets optimised, whether that is a CP mode, one axis of
one mode, or a global non-linear correction. **The base class does not know
what a mode is** — it never calls `add_mode`, and it never assumes stage index
== mode index.

---

## 3. The training loop, in detail

### 3.1 `enrich()` — the run

[base.py:113](src/neurom/training/base.py:113)

```python
self.model.train()
while self.should_add_stage(len(self.history.stages)):
    i = len(self.history.stages)
    self.prepare_stage(i)          # strategy
    record = self.stage(i)         # inner loop
    self.on_stage_end(record)      # strategy: diagnostics
    self.history.append(record)    # fills record.gain
    if record.diverged:
        self.history.stop_reason = "diverged"
        break
self.progress.close()
return self.history
```

Three properties worth knowing:

* **It is resumable.** The next stage index is `len(self.history.stages)` and
  the history accumulates on `self.history`, so calling `enrich()` a second
  time on the same trainer continues from the current state rather than
  restarting.
* **A divergent stage aborts the whole run**, not just the stage.
* **It leaves the model in training mode.** Call `model.eval()` before
  evaluating at arbitrary points, or use `pgd.evaluate(coords)` directly
  (which is what the example does).

### 3.2 `stage()` — the inner loop

[base.py:139](src/neurom/training/base.py:139)

```
loop:
    loss = step()                       # PRE-update loss
    record.losses.append(loss)
    progress.update(...)
    if not finite(loss):  -> diverged, return
    reason = stage_criterion.stop_reason(record.losses)
    if reason:
        record.stop_reason = reason
        record.final_energy = loss(model())   # one extra forward
        if not finite(final_energy): -> diverged
        return record
```

**The off-by-one that matters.** `optimizer.step(closure)` returns the loss the
closure computed *before* the update it then applied. So `record.losses[-1]`
describes the state the stage was in one update before it ended. Anything
comparing energies across stages must not use it — hence `final_energy`, one
extra forward pass after the loop, and `StageRecord.energy` preferring it
([history.py:49](src/neurom/training/history.py:49)).

**Divergence is checked twice** for the same reason: once on every recorded
(pre-update) loss, and once on `final_energy`, which is the only thing that can
catch a stage whose *last* update is what blew the model up.

`final_energy` is computed with `.detach()`, deliberately **not** under
`torch.no_grad()`: this energy differentiates internally via
`jacobian_field` → `torch.autograd.grad`, which needs a live graph. Under
`no_grad()` it raises "element 0 of tensors does not require grad"
([base.py:236](src/neurom/training/base.py:236) has the full story).

### 3.3 `step()` and the closure

[base.py:188](src/neurom/training/base.py:188)

```python
def step(self):
    return float(self.optimizer.step(self._closure()).detach())
```

Closure-based so LBFGS (which re-evaluates the loss several times per step)
works without any branching.

The closure does `zero_grad → model() → loss → backward(retain_graph=True)`.
**`retain_graph=True` is required, not defensive.** `QuadratureContext` builds
`x_phys`/`xi_back` once at construction from the `requires_grad` leaf
`_xi_ref`, and `NeuROMModel.forward` never calls `update_contexts()`, so every
iteration reads through that same pre-built subgraph. A non-retaining backward
frees it and iteration 2 dies with "Trying to backward through the graph a
second time". The alternative (rebuild contexts every forward) also works and
costs a re-map per context per iteration; revisit if the meshes ever become
trainable (r-adaptivity).

### 3.4 One fresh optimizer per stage

`make_optimizer()` ([base.py:86](src/neurom/training/base.py:86)) builds a new
optimizer over the currently-trainable parameters, and raises `RuntimeError` if
nothing is trainable (which means `prepare_stage` left everything frozen).

Why fresh: Adam's moment buffers live in `optimizer.state` indefinitely. For
purely greedy training it barely matters (a frozen parameter gets no gradient
and `Adam.step` skips it). For any strategy that *unfreezes* earlier modes —
`SimultaneousTrainer` — reusing the optimizer would resume those modes with
second-moment estimates accumulated against the energy at a **different
truncation order**. A fresh optimizer makes that impossible rather than
something to remember.

Default: `Adam(lr=0.1)`. Override with `optimizer_factory=lambda params: ...`.

---

## 4. The two strategies

Both are ~30 lines. The entire difference is one missing `freeze_all()`.

### `GreedyTrainer` — [greedy.py](src/neurom/training/greedy.py)

```python
def prepare_stage(self, stage_index):
    if stage_index > 0:
        self.decomposition.freeze_all()
        self.decomposition.add_mode()
    self.make_optimizer()
```

* Stage 0 adds nothing — it trains whatever `n_modes_ini` modes the
  decomposition was built with.
* Every later stage freezes **everything** then activates one new mode
  (`add_mode` unfreezes only what it adds).
* Classical progressive-Galerkin PGD: each mode minimises the energy over the
  residual left by its predecessors, and an error *inside* an earlier mode is
  permanent.
* `on_stage_end` records `amplitude` (of the newest mode) and
  `max_correlation` (newest against its predecessors — nothing else can move).

### `SimultaneousTrainer` — [simultaneous.py](src/neurom/training/simultaneous.py)

```python
def prepare_stage(self, stage_index):
    if stage_index > 0:
        self.decomposition.add_mode()
    for mode in range(self.decomposition.n_modes_truncated):
        self.decomposition.unfreeze_mode(mode)
    self.make_optimizer()
```

* No `freeze_all()`. Stage *m* solves the full rank-*m+1* problem, using the
  previous stage's modes as a starting point rather than a fixed background.
* Strictly the better minimiser at equal rank (greedy's answer is feasible for
  it), at the cost of re-optimising everything every stage.
* The explicit unfreeze loop is **not** redundant with `add_mode`: the
  decomposition may arrive frozen (from a previous greedy run), so the stage
  states its own requirement.
* `on_stage_end` uses `max_pairwise_correlation` — over *every* pair of active
  modes. With nothing frozen, two *earlier* modes can drift together long after
  either was added, which a newest-mode-only measure cannot see.

### Two consequences of `SimultaneousTrainer` to expect

1. **Enrichment gains are not comparable to greedy's.** A stage improves the
   energy both by adding a mode *and* by correcting the old ones, so
   `RelativeGain` sees the sum and keeps enriching slightly longer at the same
   `tol`.
2. **Stage length is not shareable between the two strategies.** Measured, in
   the `CHANGELOG`: at greedy's `min_iter=120` a simultaneous stage spends its
   budget re-fitting old modes and the new one never takes off (amplitude 3e1
   against mode 0's 3e5), small enough that `RelativeGain` calls the run
   converged after two stages. It needs 300. Do not retry a shared budget.

### `n_modes_ini > 1` buys nothing (either strategy)

Every initially-active mode starts from the same `Axis.init_values` seed and
sees the same gradient throughout stage 0, so they stay parallel forever
(measured: `max_correlation = 1.0`, amplitudes matching to 4 significant
digits). Modes only differentiate by being added at *different* stages, from
different states. Use `n_modes_ini=1`.

---

## 5. Stopping — the two criteria

Two protocols, because the two decisions see different data: a **stage**
criterion watches a stream of per-iteration losses; an **enrichment** criterion
watches completed stages. Both return a short reason string (which lands in the
history) or `None` to continue. They are injected rather than overridden as
methods, so the stopping rule and the strategy stay independent — otherwise
every combination needs its own subclass.

### Stage criteria (when does *this* stage end)

**`RelativeChange(tol=1e-4, window=20, max_iter=1000, min_iter=100, floor=1.0)`**

Improvement over a sliding window: `(past - current) / max(|current|, |past|, floor)`,
compared to `tol`. Three details are load-bearing, all learned from the beam
problem:

* **`min_iter`** — Adam has a long sticky early phase (~100 iterations) where
  the energy barely moves before it escapes and dives. Without a floor on the
  iteration count, a plateau detector calls that convergence.
* **`floor`** — the energy crosses zero (~+2e5 to ~-1e8) and spans eight orders
  of magnitude, so a purely relative denominator blows up near the crossing.
* **signed improvement** — a *rising* loss reads as no progress and stops the
  stage, rather than looking like a large change.

**`FixedIterations(n_iter)`** — exactly `n_iter` iterations. Fully
reproducible, blind to whether anything converged. Reproduces the pre-trainer
fixed-length loop.

Both implement `budget()`, which is **advisory only**: nothing in the loop
reads it, it is what lets `ProgressBar` show a fraction instead of a bare
counter. A criterion with no hard cap returns `None` and the bar degrades to a
counter rather than lying.

### Enrichment criteria (do we start another stage)

**`RelativeGain(tol=1e-4, floor=1.0)`** — compares the last two stages'
`energy`; stops when the relative gain falls below `tol`. Returns `None` for
fewer than two stages.

**`MaxStages(n_stages)`** — a hard stage budget.

### Where the run's `stop_reason` comes from

`should_add_stage` sets `self.history.stop_reason` before returning False. Both
concrete strategies check **capacity first**:

```python
if stage_index == 0: return True                       # stage 0 always runs
if n_modes_truncated >= n_modes_max:
    history.stop_reason = "capacity"; return False
if (reason := enrichment_criterion.stop_reason(history.stages)):
    history.stop_reason = reason; return False
return True
```

So a run satisfying both on the same call reports `"capacity"`. Stage 0 returns
True unconditionally — which is why `MaxStages(0)` still gets exactly one stage
in (stage 0 adds no mode, it trains the already-active ones).

Possible values: run-level `"capacity"`, `"converged"`, `"n_stages"`,
`"diverged"`; stage-level `"converged"`, `"max_iter"`, `"n_iter"`,
`"diverged"`.

---

## 6. The history

[history.py](src/neurom/training/history.py) — plain dataclasses, no torch
inside, which is what makes them safe to pickle into a checkpoint.

**`StageRecord`**

| field | meaning |
|---|---|
| `stage` | index within the run |
| `losses` | every iteration's loss, each **pre-update** |
| `stop_reason` | why the stage ended |
| `diverged` | a non-finite loss ended it |
| `gain` | improvement over the previous stage's energy; NaN for stage 0; filled by `TrainingHistory.append` |
| `final_energy` | loss of the state the stage actually ended in |
| `diagnostics` | strategy-specific dict; base never writes here |
| `n_iter` (property) | `len(losses)` |
| `energy` (property) | `final_energy` if set, else `losses[-1]`, else NaN |

`energy` prefers `final_energy` for the off-by-one reason in §3.2. A genuinely
NaN `final_energy` is returned as-is — never mistaken for "not computed".

**`TrainingHistory`** — `stages` (list) + `stop_reason`, plus
`losses` (all stages concatenated, for plotting) and `append(record)` which
fills in the gain.

---

## 7. Diagnostics — is the decomposition degenerate?

[diagnostics.py](src/neurom/training/diagnostics.py). Every mode is seeded with
the same `Axis.init_values`, so nothing in the optimisation *guarantees* two
modes end up different. Freezing is what should prevent it in the greedy
strategy — but that is a hope until measured.

* `monom_values(decomposition, mode)` → the mode's full nodal values per axis.
  Uses `full_values()` (constrained DOFs included), which is correct **only
  because the Dirichlet data here is homogeneous**. With inhomogeneous
  Dirichlet data every mode would share the same nonzero component on those
  DOFs and every correlation below would be biased upward.
* `amplitude(values)` = `prod_k ||w^k||` — the mode's overall size.
* `correlation(a, b)` = `|prod_k <w_a^k, w_b^k> / (||w_a^k|| ||w_b^k||)|` — 1
  when the two rank-1 tensors are parallel on every axis, i.e. copies up to the
  CP scale invariance. A zero-norm axis reports 0.0 (a vanishing mode
  duplicates nothing); NaN propagates.
* `max_correlation(d, mode)` — largest against modes `0..mode-1`. Greedy's.
* `max_pairwise_correlation(d)` — largest over every pair. Simultaneous's.

`_largest` exists because `max(0.0, nan) == 0.0` in Python — it would report
perfect orthogonality exactly when the state is garbage. It propagates NaN
instead.

**Caveat shared by all of them:** raw nodal vectors, not the
quadrature-weighted L2 inner product. This is *not* the energy-norm
correlation. Cheap enough to spot duplication, not to quantify it.

**How to read them:** a tiny `gain` together with `max_correlation` near 1 is
the sequence rediscovering a mode it already has. That is the failure mode
recorded in memory: below ~80–120 iterations per stage, greedy stages simply
re-find mode 0.

---

## 8. Progress reporting

Injected, so the loop stays silent by default (tests, batch scripts) and a
terminal run can show a bar without either knowing about the other. No tqdm
dependency — the bar is thirty lines and writes to **`sys.stderr`**, so a
redirected stdout carries only results.

* `ProgressReporter` — the no-op default and the interface:
  `stage_start(i, budget)`, `update(iteration, loss)`, `stage_end(record)`,
  `close()`.
* `ProgressBar(stream=None, width=None, every=0.1)` — one line per stage,
  redrawn in place, finalised on `stage_end` with the stop reason so finished
  stages stay readable above the running one:

```
stage 2 |=========      |  180/600  E=-2.0170e+11   12.4s
stage 2 |===============|      203  converged  E=-2.0170e+11   14.0s
```

  `every` throttles redraws (redrawing every iteration of a cheap stage costs
  more than the iteration), but iteration 1 always draws — otherwise a stage
  ending inside the throttle window shows nothing and the run looks hung.

Write your own by implementing the four methods (e.g. a CSV logger, a live
matplotlib window).

---

## 9. Stopping, saving and reloading

### Stopping a run

There is no interrupt handler. A run ends because:

1. the enrichment criterion fires (`RelativeGain`, `MaxStages`),
2. the decomposition hits `n_modes_max` (`"capacity"`),
3. a stage diverges (non-finite loss), which breaks the loop immediately,
4. `Ctrl-C` — nothing is saved; the checkpoint is written by the *caller*
   after `enrich()` returns.

To bound a run in advance: `MaxStages(n)` for the number of stages,
`RelativeChange(max_iter=...)` or `FixedIterations(n)` for the length of each.

### Saving

[checkpoint.py](src/neurom/training/checkpoint.py)

```python
save_checkpoint(path, model, history=None, metadata=None)
```

Writes `{"format": 1, "state_dict": model.state_dict(), "history": history,
"metadata": {...}}` via `torch.save`.

**Why a `state_dict` is enough.** The truncation needs no bookkeeping of its
own: `active` is a registered **buffer** on every `QuadratureAssembly`, so how
many modes were active round-trips with the values themselves. There is no way
for the two to disagree.

**What is deliberately *not* saved:** the geometry — meshes, shape functions,
quadrature rules, the energy. A checkpoint is meaningless without the code that
built the model. `metadata` is where the caller records what that model was
(mesh sizes, strategy, criteria) so a mismatch can be *caught* rather than
silently plotted.

### Reloading

```python
history, metadata = load_checkpoint(path, model, strict=True)
```

Takes a **freshly built model** and fills it in place. Same axes, same mesh
sizes, same `n_modes_max` — a mismatch surfaces as a shape error from
`load_state_dict`, which is intended: silently loading half a decomposition
would produce plots that look plausible and are wrong. A different `format`
version raises `ValueError`.

Uses `weights_only=False` because the payload carries the `TrainingHistory`
dataclass, not only tensors. **Only point it at checkpoints you wrote.**

**The freeze state is not restored** — `requires_grad` is not part of a
`state_dict`. It does not need to be: `prepare_stage` rewrites the whole freeze
state at the start of every stage, so resuming training from a checkpoint is
well defined regardless of how the parameters arrive. What *is* restored is
which modes are active.

### Resuming training from a checkpoint

```python
problem = build_problem(...)                       # same geometry
history, meta = load_checkpoint(path, problem.model)

trainer = GreedyTrainer(problem.model, stage_criterion=..., enrichment_criterion=...)
trainer.history = history        # so stage indices and RelativeGain continue
history = trainer.enrich()       # picks up at stage len(history.stages)
save_checkpoint(path, problem.model, history, metadata=meta)
```

Assigning `trainer.history` is what makes it a *resume* rather than a restart:
`enrich` derives the next stage index from it, and `RelativeGain` reads it. The
active-mode count comes from the checkpoint's `active` buffers, so
`prepare_stage` adds mode *n+1*, not mode 1.

---

## 10. Minimal end-to-end usage

```python
from neurom.training import (
    GreedyTrainer, SimultaneousTrainer,
    RelativeChange, RelativeGain, MaxStages, FixedIterations,
    ProgressBar, save_checkpoint, load_checkpoint,
)

trainer = GreedyTrainer(
    model,                                  # NeuROMModel
    stage_criterion=RelativeChange(tol=1e-3, window=20, max_iter=600, min_iter=120),
    enrichment_criterion=RelativeGain(tol=1e-3),
    progress=ProgressBar(),                 # omit for silence
    # optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.05),
)
history = trainer.enrich()

print(history.stop_reason)
for r in history.stages:
    print(r.stage, r.n_iter, r.stop_reason, r.energy, r.gain,
          r.diagnostics["amplitude"], r.diagnostics["max_correlation"])

save_checkpoint("run.pt", model, history, metadata={"strategy": "greedy"})
```

---

## 11. Extending it — where the seams are

| You want | Override / inject |
|---|---|
| A different freeze schedule (alternating directions, block-wise) | `prepare_stage` |
| A different enrichment rule | `should_add_stage`, or just an `EnrichmentCriterion` |
| A different inner iteration (axis sweeps) | `step` — reuse `self._closure()`, do **not** copy the `retain_graph` handling |
| A different inner loop entirely | `stage` |
| New per-stage measurements | `on_stage_end`, writing into `record.diagnostics` |
| A different optimizer | `optimizer_factory` |
| A different plateau rule | a `StageCriterion` subclass — implement `stop_reason(losses)` and, if you can, `budget()` |
| Different output | a `ProgressReporter` subclass |

For a strategy whose stage is *not* a mode (the non-linear PGD case), note that
nothing in `base.py` assumes otherwise: `StageRecord.stage` is a stage index,
the trainable set is read off `requires_grad`, and only `on_stage_end`'s
diagnostics currently speak of modes.

The one caveat on overriding `step`: the base's `prepare_stage` contract says
the optimizer must exist by the time the first `step` runs — usually via
`make_optimizer()` at the end of `prepare_stage`, but a strategy building one
optimizer per axis may arrange it differently.

---

## 12. Tests

* `tests/unit/training/` — `test_base.py`, `test_criteria.py` (implied),
  `test_checkpoint.py`, `test_diagnostics.py`, `test_progress.py`,
  `test_simultaneous.py`. The base tests use a stub loss with no internal
  autograd — which is exactly why the `no_grad()` bug in `_record_final_energy`
  only surfaced on real data.
* `tests/integration/test_greedy_trainer.py` — the real energy.
* `docs/examples/1d_5-parametric_beam_PGD/tests/` — the example's own tests,
  including `test_checkpoint_round_trip.py` and `test_simultaneous_trainer.py`.
