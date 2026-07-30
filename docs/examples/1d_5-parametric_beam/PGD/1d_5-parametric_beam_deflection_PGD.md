# The 5-parametric bar — how the example works, end to end

Companion to [`1d_5-parametric_beam_deflection_PGD.py`](1d_5-parametric_beam_deflection_PGD.py)
and [`reference_fem_solution.py`](reference_fem_solution.py). For the training
machinery itself see [`docs/TRAINING_MODULE_GUIDE.md`](../../TRAINING_MODULE_GUIDE.md).

---

## 1. TL;DR — how to run it

```bash
# once, before anything can be scored: the FEM ground truth
python docs/examples/1d_5-parametric_beam_PGD/reference_fem_solution.py
```

```bash
# train (or reload) and plot, greedy schedule
python docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py greedy
```

```bash
# the other schedule, retraining from scratch and overwriting its checkpoint
python docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py simultaneous --retrain
```

CLI grammar: `python <file> [greedy|simultaneous] [--retrain]`. Default
`greedy`. **Without `--retrain`, an existing `pgd5_<strategy>.pt` is loaded and
only the figures are redrawn** — iterating on a plot costs seconds, retraining
costs minutes. One checkpoint per strategy, so the two runs never overwrite each
other.

From Python:

```python
from neurom.training import SimultaneousTrainer
problem = main(trainer_cls=SimultaneousTrainer, plot=False)   # trains or loads
problem = main(train=False)                                   # assemble only
```

---

## 2. The physics

A bi-clamped 1-D bar of length 10 under a constant axial load, with a
tanh-graded modulus:

$$E(x, E_1, E_2, \alpha, n) = \frac{E_2 - E_1}{2}\tanh\!\big(n(x-\alpha)\big) + \frac{E_2 + E_1}{2}$$

Because $E_1, E_2 > 0$ and $\tanh$ maps into $(-1,1)$, the modulus stays
strictly between $E_1$ and $E_2$ — **positivity is structural**, no clamping
needed.

The four material/geometry parameters become extra coordinates, so the solution
is sought in separated form:

$$u(x, E_1, E_2, \alpha, n) = \sum_{i=1}^{m} X_i(x)\,\lambda_i(E_1)\,\mu_i(E_2)\,A_i(\alpha)\,N_i(n)$$

The energy minimised is `elastic + load` (the load field carries its own sign),
matching the 2-parameter example's convention.

### Discretisation

| axis | interval | nodes | constraint |
|---|---|---|---|
| `space` | [0, 10] | 30 | `Dirichlet` at both ends, value 0 |
| `E1` | [10, 100] | 20 | `NoConstraint` |
| `E2` | [10, 100] | 20 | `NoConstraint` |
| `alpha` | [2, 8] | 15 | `NoConstraint` |
| `n` | [0.5, 5] | 15 | `NoConstraint` |

`LinearSegment` shape functions, `IsoparametricMapping1D`, `MidPoint1D`
quadrature (injectable — the tests pass `TwoPoints1D` to catch broadcasting bugs
that a single quadrature point hides). Load = 1000.0. `N_MODES_MAX = 10` (the
2-parameter case was exactly rank-1; a non-separable modulus has no reason to be
low-rank). `alpha` is kept away from the clamped ends so the transition always
sits inside the bar; `n` spans near-uniform to a sharp interface roughly one
length unit wide.

> **`AXIS_ORDER` is load-bearing.** It fixes the column order of
> `CPPGD.evaluate` and the key order of `CPPGD.directory()`. Every query matrix
> built anywhere in the example stacks columns in that order.

---

## 3. Assembly — `build_problem`

```python
Problem = build_problem(loss_fn, *, n_modes_max, n_modes_ini, n_nodes, quad)
```

Returns a `Problem` dataclass: `model`, `pgd`, `field_layout`, `domain`, `axes`,
`history`.

The wiring, in order:

1. `make_axis(...)` × 5 — each builds a uniform `Topology`, a `Field` of nodal
   positions, and an `Axis` (which builds its own `Mesh` and `QuadratureContext`
   in `__post_init__`). Every new monom is seeded with the constant
   `init_value=0.5`.
2. `CPPGD(axes, n_modes_max=10, n_modes_ini=1)` — the grid of
   `n_modes_max × n_axes` `TrainableField` monoms, all registered up front, with
   only the leading `n_modes_ini` mode-blocks *active*.
3. The **load** is a `Field` f(x) on the space axis's topology, interpolated on
   the **same quadrature** as the space monoms so `inner(f, X)` aligns
   point-for-point. It is the single spatial factor of the separated source
   $f = f_0(x)\otimes 1(E_1)\otimes\cdots$; the constant parametric factors are
   carried by the $\int\lambda_i\,dE_1$ terms in the load part of the energy.
4. `IntegrationDomain([*pgd.assemblies(), assembly_f])` — one shared domain
   interpolating everything the energy reads.
5. `NeuROMModel(field_layout, pgd, domain, loss=lambda layout: loss_fn(layout, pgd))`.

**The energy is injected**, not referenced: `build_problem` never mentions
`energy` by name, so the same wiring drives a different (e.g. non-linear PGD)
functional unchanged. That is the seam this branch exists to use.

---

## 4. The separated energy — `energy(field_layout, decomposition)`

The one interesting function. Every factor of the elastic term separates into a
product of 1-D integrals **except** $\tanh(n(x-\alpha))$, which couples three
axes.

For each mode pair $(i,j)$ it computes:

| quantity | meaning |
|---|---|
| `Kx` | $\int X_i' X_j'\,dx$ |
| `L0`, `L1` | $\int \lambda_i\lambda_j\,dE_1$ and $\int E_1\lambda_i\lambda_j\,dE_1$ |
| `M0`, `M1` | the same on $E_2$ |
| `P0`, `Q0` | $\int A_iA_j\,d\alpha$, $\int N_iN_j\,dn$ |
| `T` | the coupled block |

and assembles

```
elastic += mean_modulus * Kx * P0 * Q0  +  half_contrast * T
mean_modulus  = 0.5 * (M1*L0 + L1*M0)      # (E2 + E1)/2
half_contrast = 0.5 * (M1*L0 - L1*M0)      # (E2 - E1)/2
```

then halves the total. The load term is fully separable — one 1-D integral per
factor.

### The coupled block

`tanh_grid` is $\tanh(n_q(x_q-\alpha_q))$ on the **tensor product** of the
space, alpha and n quadrature points, shape `(Qx, Qalpha, Qn)`, contracted by a
single `einsum("xan,x,a,n->", ...)` against the three raw densities. It does not
depend on the mode pair, so it is built once per call — and rebuilt every call
rather than cached at setup, so it stays correct if the meshes ever become
trainable (r-adaptivity).

### Two traps documented in the source

* **`jacobian_field` returns one extra trailing axis** `(N_e, N_q, *u_shape, d)`
  which must be *contracted away* by `inner()`, not reshaped away. Reshaping
  only appears to work for `d = 1` and silently turns the cross terms into an
  element-wise broadcast. (`kx_density`, `a_density`, `n_density` *are* safe to
  `.reshape(-1)` — they are scalar `(N_e, N_q, 1)` densities with no trailing
  vector axis.)
* **Cross terms assume modes `i` and `j` share a mesh per axis** — the measure
  and coordinates indexed by `i` are used for both. `tanh_grid` is the more
  fragile consumer, since it hard-codes mode 0's quadrature points for every
  pair. Independent per-mode meshes would need a common intersection mesh with a
  recomputed measure.

---

## 5. `main()` — the driver

```python
main(verbose=True, train=True, plot=True,
     trainer_cls=GreedyTrainer, stage_min_iter=None,
     checkpoint=None, retrain=False)
```

Flow:

1. **Always** build the problem and evaluate the energy once at the untrained
   state. This proves the whole chain (five axes → CP-PGD → shared
   `IntegrationDomain` → separated energy) assembles and produces a finite
   value. `train=False` stops here and returns the untrained `Problem`.
2. Resolve `checkpoint` (default `checkpoint_path(trainer_cls)` →
   `pgd5_greedy.pt` / `pgd5_simultaneous.pt` next to the script) and
   `stage_min_iter` (default `STAGE_MIN_ITER[trainer_cls]`).
3. **If the checkpoint exists and `retrain` is False → load instead of train.**
   Prints the saved strategy and warns if the saved mesh sizes differ from
   `DEFAULT_N_NODES` (a shape mismatch would have raised; a *node-count*
   mismatch cannot, since it is baked into the shapes — but the criteria or
   strategy may still differ and the plots would silently describe the old run).
4. Otherwise construct the trainer, `enrich()`, and `save_checkpoint` with
   metadata `{strategy, n_nodes, stage_min_iter, n_modes}`.
5. `_report(...)` — **shared by both paths**, so a checkpoint produces exactly
   the same output as the run that wrote it. Otherwise the two paths drift and
   "plot from the checkpoint" stops being a faithful shortcut.

### The trainer configuration, and why

```python
trainer_cls(
    problem.model,
    stage_criterion=RelativeChange(
        tol=1e-3, window=20,
        max_iter=max(600, 2 * stage_min_iter),
        min_iter=stage_min_iter,
    ),
    enrichment_criterion=RelativeGain(tol=1e-3),
    progress=ProgressBar() if verbose else None,
)
```

`min_iter` is load-bearing **twice**:

* Adam spends a long sticky early phase on this energy where the loss barely
  moves, which a plateau detector reads as convergence.
* Below ~80 iterations per stage the greedy step simply **rediscovers mode 0** —
  `max_correlation` reads 1.0 and the "modes" are copies of each other. The
  printed *max corr* column is what tells you whether that is happening.

It is also the one setting the two strategies must **not** share:

```python
STAGE_MIN_ITER = {GreedyTrainer: 120, SimultaneousTrainer: 300}
```

A greedy stage spends its whole budget on one new mode; a simultaneous stage has
to re-fit every earlier mode as well, and at 120 iterations its new mode never
takes off (amplitude 3e1 against mode 0's 3e5; gain 1.7e7 against greedy's
5.6e9) — small enough that `RelativeGain` calls the run converged after two
stages. At 300 it takes off and overtakes greedy. Measured, not guessed; see the
`CHANGELOG`.

`ProgressBar` is on only when `verbose`, so scripted runs and tests print
nothing. A stage of this problem is minutes long — a silent run is
indistinguishable from a hung one.

---

## 6. The output

### The per-stage table

```
training stopped: converged
stage  iters       stop         energy         gain   amplitude  max corr
    0    203  converged  -1.874321e+11   nan         3.1e+05      0.000
    1    600   max_iter  -2.017044e+11   1.43e+10    8.4e+04      0.312
    ...
```

* `energy` is `final_energy` — the state the stage *ended in*, not the
  pre-update `losses[-1]`.
* `gain` is the improvement over the previous stage; NaN for stage 0.
* `amplitude` = $\prod_k \|w^k\|$ of the newest mode.
* `max corr` — for greedy, the newest mode against its predecessors; for
  simultaneous, the worst **pairwise** correlation over all active modes (with
  nothing frozen, two *earlier* modes can collapse long after either was added).

**Read a tiny `gain` together with `max corr` near 1 as the sequence
rediscovering a mode it already has.** An `amplitude` near the seed size (~1)
means the stage added a mode that never grew, and its gain came from re-fitting
the existing ones.

### The three figures

| function | file | what it shows |
|---|---|---|
| `plot_convergence` | `pgd5_convergence.png` | energy vs iteration, all stages concatenated, with a dotted rule at every stage boundary |
| `plot_solution` | `pgd5_vs_reference.png` | two columns, one per highlighted reference point: `E(x)` on top, `u(x)` reference-vs-PGD below, pointwise error on a twin axis |
| `plot_modes` | `pgd5_modes.png` | every mode's factor on every axis, each normalised by its max modulus |

`plot_modes` normalises because **a CP mode is defined only up to a per-axis
scale** — multiply one factor by `c`, divide another by `c`, and the product is
unchanged. Only the *shape* is meaningful. Two modes whose curves coincide on
every axis are the degenerate case `max_correlation` reports.

`plot_solution` draws two points rather than all twelve because ten superposed
near-parabolas say very little. The default pair is `metadata["highlight"]` from
the reference bundle: sharp, high-contrast moduli where `u` has a visible kink
and a separated representation is under real strain. The twin-axis pointwise
error is there because a small global L2 can hide a local failure right at the
modulus transition — exactly where this problem is hard.

---

## 7. The reference solution and the two error measures

### Generating it — `reference_fem_solution.py`

One plain, **non-reduced** `FEMModel` per parameter point, minimising the *same*
energy by LBFGS, with the modulus injected **analytically at the quadrature
points** (no interpolation error on E). Key settings:

* `DTYPE = float64`. In float32 the LBFGS solution stalls near 1e-3 relative
  (measured: halving both moduli, which must scale `u` by exactly 2, was off by
  3e-3). In float64 that becomes 8e-5, i.e. the reference stops being the
  limiting error. The bundle is cast back to the caller's dtype on load. The
  scoped `double_precision()` context manager exists because importing the
  example sets the global default dtype to float32.
* `N_NODES = 400`, `TwoPoints1D` — much finer than the PGD's 30-node space axis,
  and a two-point rule so the graded modulus is integrated properly inside each
  element (with `n = 5` the midpoint rule is visibly off).
* A full tensor grid, `N_GRID = 5` inclusive points per parameter axis
  (`grid_parameters`), so **5×5×5×5 = 625** points — 5 in `n`, 5 in `alpha`, and
  25 in the `E1`–`E2` plane at every `(alpha, n)` — plus the 2 `EXTREME_POINTS`
  = **627**, each sampled at `N_X_SAMPLES = 101` positions. The grid is what
  makes `overall` a genuine global L2 measure over the whole parameter box
  rather than a handful of random draws. (`sample_parameters`, the old random
  draw, is kept as a primitive but no longer feeds the reference set.)
* `load_example()` imports the PGD script by path, so the geometry, intervals,
  load and modulus law have a **single source of truth** and cannot drift.

The two hard points are deliberately *not* mirror images — opposite contrast
direction, different ratio (10× vs 5×), transition on either side of mid-span,
so nothing about the second is implied by getting the first right:

```
soft-stiff-left    E1=10,  E2=100, alpha=3.0, n=5.0
stiff-soft-right   E1=100, E2=20,  alpha=6.5, n=4.0
```

Saved to `reference_solution.pt` as a dict with `x`, `params`, `param_names`,
`u`, `labels`, `metadata` (including `highlight`).

> **The modulus law lives in the example** (`modulus(x, E1, E2, alpha, n)`). The
> separated `energy` hard-codes its tanh structure to keep the integrals
> separable, and the reference module injects *this* function. If the law
> changes, both must change together.

### `relative_errors(pgd, reference=None)`

Returns `{"per_point": {label: float}, "overall": float, "u_pgd": Tensor}`. Two
different things, because they answer different questions:

* **`per_point[label]` is space-only**: $\|u_{PGD}(\cdot,p) - u_{ref}(\cdot,p)\| / \|u_{ref}(\cdot,p)\|$
  over the 101-point `x` grid with the parameters frozen at `p`. This is what
  one panel of `plot_solution` shows.
* **`overall` is space and parameter points jointly**: both `(627, 101)` tables
  are flattened and one ratio of norms is taken. It is dominated by the points
  with the largest deflection (the soft ones), which is the honest global
  figure — the *mean* of the per-point errors would weight a barely-loaded stiff
  bar as much as a soft one.

`overall` still does not weight the parameter volume — the flattened norm treats
every grid point equally — but the `N_GRID = 5` inclusive tensor grid is a
uniform sweep of the box, so it now reads as a global measure over all five axes
rather than the average of a handful of random draws.

---

## 8. Checkpoints

`pgd5_greedy.pt` / `pgd5_simultaneous.pt`, written next to the script,
**gitignored** (a few hundred kB of derived data, regenerated by one command).

Each holds `model.state_dict()` + the `TrainingHistory` + metadata. That is
enough because `active` is a registered buffer on every `QuadratureAssembly`, so
which modes were active round-trips with the values — a loaded model evaluates
exactly what the run produced. The history rides along too, which is what lets
the convergence plot and the diagnostics table be redrawn without retraining.

Geometry is *not* saved: `load_checkpoint` fills a freshly built model, and a
mismatch raises from `load_state_dict` rather than silently plotting half a
decomposition.

`checkpoint=False` disables the mechanism entirely (train every time, save
nothing) — what the tests use.

---

## 9. Measured results (2026-07-24, default meshes)

Scored on `reference_solution.pt`, same criteria, 300-iteration stages:

| | rank | overall | worst point | energy |
|---|---|---|---|---|
| simultaneous | 8 | **2.24e-2** | 3.52e-2 | -2.0522e11 |
| greedy | 8 | 4.17e-2 | 1.15e-1 | -2.0451e11 |

Under its own `RelativeGain(1e-3)` stop, simultaneous takes 7 stages for
2.66e-2 / 5.69e-2 — better than greedy's 8-stage 3.11e-2 / 6.13e-2 with one mode
fewer.

**Open, unexplained:** even at 300 iterations, roughly every other simultaneous
stage adds a mode that stays at seed size (amplitude ~1) and gets its gain from
re-fitting the existing modes. And greedy at rank 8 scores *worse* against the
reference than at rank 6 (4.17e-2 vs 3.21e-2) despite a lower energy — **lower
energy is not monotonically better accuracy at these sampled points.**

---

## 10. Tests

In [`tests/`](tests/), next to the example rather than in the repo's test tree:

* `test_1d_5_parametric_beam_energy.py` — the separated energy (this is where
  the `TwoPoints1D` injection matters).
* `test_greedy_trainer.py`, `test_simultaneous_trainer.py` — the two schedules
  on a tiny problem (`n_nodes=` override).
* `test_reference_fem_solution.py` — the FEM reference, including the
  analytical check (`check_against_analytical`) and the scaling identity.
* `test_checkpoint_round_trip.py` — save → rebuild → load → identical
  evaluation.

---

## 11. Extending it

The injection points, in order of how likely you are to want them:

| To change | Touch |
|---|---|
| the functional (→ non-linear PGD) | pass a different `loss_fn` to `build_problem`; nothing else |
| the schedule | `trainer_cls=` in `main`, or add to `STRATEGIES` |
| the stopping rules | the `RelativeChange` / `RelativeGain` construction in `main` |
| mesh sizes | `DEFAULT_N_NODES` (and regenerate the checkpoints) |
| quadrature | `quad=` on `build_problem` |
| the modulus law | `modulus()` **and** the tanh block in `energy()`, together |
| the reference points | `EXTREME_POINTS` / `N_GRID` (grid density per axis), then regenerate |

Note that a change to the geometry or mesh sizes **invalidates every existing
checkpoint** — `load_state_dict` will raise on shape, which is the intended
outcome, but the `n_nodes` warning in `main` is what catches the cases it
cannot.
