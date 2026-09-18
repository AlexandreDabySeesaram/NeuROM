# CP-PGD implementation — session handoff notes

> Read this file to catch up on the CP-PGD work. It is a self-contained summary;
> the design spec and implementation plan it links carry the full detail. The
> body reflects the latest `FieldLayout`/`PGDFEMModel` integration; the original
> CP-PGD module landed earlier on branch `develop_solal`.

> **⚠️ Historical log.** The dated sections below are a point-in-time record and
> some names are now superseded: `PGDFEMModel` → `NeuROMModel`
> (`src/neurom/neurom_model.py`), and `separated_view` / `interpolate_separated`
> were removed (read monoms back by name from the `FieldLayout` via
> `directory()`, or with `PointWiseInterpolator` at arbitrary points). The
> "Evaluation" section immediately below is kept current.

## Evaluation: diagonal (`evaluate`) vs grid (`assemble`)

Two ways to sample the trained field, with **different input formats and
different semantics**. Pick by whether you want *paired* points or *every
crossing*:

| | `evaluate` (= `NeuROMModel.forward`, eval mode) | `assemble` (= `NeuROMModel.assemble`) |
|---|---|---|
| **input** | one `(P, n_axes)` tensor, **a point per row** (`pts[p] == (x_p, E_p, …)`) | a **list** of one 1-D tensor per axis |
| **axis lengths** | all equal to `P` | **independent** `(N_1, …, N_l)` |
| **computes** | the `P` **paired** tuples (diagonal) | **every** combination (tensor product) |
| **output** | `(P, d)` | `(N_1, …, N_l[, d])` |

- **Diagonal / `evaluate`** — a cloud of arbitrary query points, or a 1-D slice
  (fix `E`, sweep `x` by pairing each `x` with the same `E`). Because it pairs
  columns element-wise, every axis column must share the length `P`.
- **Grid / `assemble`** — a full parametric surface / heatmap over `x` × `E`;
  the lengths per axis are free and it returns the whole `(N_x, N_E)` tensor.

Internals: both interpolate each monom on its axis with `PointWiseInterpolator`,
then combine. `evaluate` multiplies the per-axis columns and sums the modes;
`assemble` builds a dynamic `einsum` (mode index summed, one grid letter per
axis, plus a component letter for a vector axis). The trailing `d` is the single
vector factor's dim, dropped when all factors are scalar. Both detach.

> **API note (2026-07-15):** `evaluate` takes the `(P, n_axes)` **row tensor**
> (one point per row) — the human-friendly form. It no longer accepts the old
> list-of-one-1-D-tensor-per-axis format; `assemble` still uses that list form.

**Date:** 2026-07-10
**Branch:** `pgd_addition_solal`
**Commit range:** `1208881` → `e2fbd4f` (FieldLayout/PGDFEMModel integration)
**Test status:** full suite `uv run pytest` → **79 passed**

## What this adds

A CP-PGD (canonical-polyadic Proper Generalized Decomposition) module: a pure
separated-representation model for

```
u({x_k}_{k=1..l}) = Σ_{m=1..M}  Π_{k=1..l}  w_m^k(x_k)
```

generalized to an arbitrary number of axes `l` (`w_m^k` = a **monom**, the
product over `k` = a **mode**), with greedy mode enrichment. Built on the
existing modular `neurom` FE building blocks (`TrainableField`, `Mesh`,
`QuadratureContext`, `QuadratureAssembly`, `PointWiseInterpolator`).

**Key architectural decision:** the energy/loss and the training loop live
**outside** the module (they are problem-specific and may later become
non-separable). The module only holds the modes, evaluates them, and manages
enrichment.

## Files

### Added — production code

- `src/neurom/decompositions/__init__.py` — exports `TensorDecomposition`,
  `FactorSpace`, `MonomSpec`, `CPPGD`, `PGDFEMModel`.
- `src/neurom/decompositions/base.py` — **`TensorDecomposition(nn.Module, ABC)`**:
  the format-agnostic contract a PGD FEM model depends on. Abstract
  `register_into(field_layout)` (register this decomposition's factor fields
  in the layout, once, at setup) and `fill(field_layout)` (re-interpolate the
  active factor fields and `update()` them in the layout; the PGD analogue of
  `IntegrationDomain.interpolate_all`, called once per forward). CP today,
  Tucker/TT later, all drive the same `PGDFEMModel` through this seam.
- `src/neurom/decompositions/factor.py` — the two descriptors:

  **`FactorSpace`** (dataclass) — the discretised space one factor lives on,
  and nothing else. Fields: `name`, `mesh`, `mapping` (which encloses the
  *geometry* shape function), `quad` (`QuadratureRule`). Builds its
  `QuadratureContext` in `__post_init__`. Read-only properties `connectivity`
  and `nodes_positions` delegate to the mesh (guaranteeing the same
  `Connectivity` object is shared by the `Mesh` and the monoms'
  `TrainableField`). Shared by every field posed on it: all the monoms of that
  factor, and any load read on the same quadrature points.

  **`MonomSpec`** (dataclass) — the blueprint for the monoms posed on a
  `FactorSpace`, **not** a monom itself. Fields: `space`, `sf` (the *field's*
  shape function, distinct from the geometry one for a sub/super-parametric
  element), `constraint` (`Constraint`, carries the BCs), `init_values` (initial
  nodal values for each monom on this factor). One spec per factor, whatever the
  number of modes: `CPPGD` reads it once per mode to build the corresponding
  `TrainableField`.

- `src/neurom/decompositions/pgd.py`:

  **`CPPGD(TensorDecomposition)`** — `CPPGD(monom_specs: list[MonomSpec], name="pgd", n_modes_ini=1)`.
  A **dynamic container**, not a training strategy: nothing is pre-allocated,
  there is no maximum number of modes, and no freezing policy is applied.
  - `self.monoms` — `ModuleList` over modes of `ModuleList` over factors of
    `TrainableField`; `self.monoms[m][k]` is the monom `w_m^k`. Grows one row
    at a time.
  - `self._contexts` — the factor spaces' `QuadratureContext`, shared across
    modes, **including modes added later**.
  - `self.n_modes` — `len(self.monoms)`. No separate counter, no active/inactive
    distinction: a mode that was never added does not exist.
  - `register_into(field_layout)` — brings the layout in phase with the monoms
    that exist now. Idempotent (`FieldLayout.add` is identity-aware), so it is
    called at setup **and again after every `add_mode()`**. This is how a new
    mode reaches the layout without the decomposition ever holding a reference
    to it.
  - `assemblies()` — the `QuadratureAssembly` of every monom, `n_modes * l` of
    them over only `l` distinct contexts. Because the list grows, it is **not**
    stored in the `IntegrationDomain`: `NeuROMModel.forward` re-reads it on
    every forward and passes it to
    `domain.interpolate_all(layout, assemblies)`. The domain holds the static
    assemblies only (the load, a source).
  - `fill(field_layout)` — for each active mode and axis, interpolates the
    monom at that axis's quadrature points and `update()`s it in the layout.
    Inactive monoms stay registered but uninterpolated (reading them raises
    `RuntimeError`, per the `FieldLayout` contract).
  - `separated_view(field_layout) -> dict[str, list[QuadratureAssemblyResult]]`
    — reads the active monoms' interpolations back out of the (already-filled)
    layout: per axis name, a list indexed by mode; entry `m` is the
    interpolation of the single monom `w_m^k` (`u` shape `(N_e, N_q, u_dim)`,
    shared `x`, `measure`). **Exposes each monom individually** so a separable
    energy can be written monom-by-monom. **Replaces** the old
    `interpolate_separated()` (removed — no `FieldLayout` bypass). Each monom
    keeps its own autograd link, so `jacobian` applies per monom.
  - `assemble(coords: list[torch.Tensor]) -> torch.Tensor` — full tensor of
    shape `(N_1, ..., N_l)` = `Σ_m Π_k w_m^k(coords[k])`, via
    `PointWiseInterpolator` per monom + a dynamically-built einsum over any
    number of axes (mode index = uppercase `Z`, axes = lowercase `a..`).
    Returns a detached tensor (for post-processing / viz / tests). Unchanged
    by the `FieldLayout` migration.
  - Enrichment: `add_mode()` **builds** a new row — one `TrainableField` and
    one `QuadratureAssembly` per `MonomSpec` — and returns its index. It is
    unbounded, it seeds the new monoms from `MonomSpec.init_values` (an
    all-zero mode is a stationary point of the energy and never takes off), and
    it freezes nothing. The caller then re-runs `register_into(layout)` and
    `NeuROMModel.add_mode_to_optimizer(optim)`.
  - Freezing utilities, mechanism without policy — `freeze_mode` /
    `unfreeze_mode` (a **row**: progressive greedy), `freeze_factor` /
    `unfreeze_factor` (a **column**: alternating-direction minimisation),
    `freeze_monom` / `unfreeze_monom` (a **cell**: classical ADM), plus
    `freeze_all` / `unfreeze_all`. The decomposition calls none of them itself.
    A frozen monom is **still part of `u`**: `requires_grad` means "not
    optimised", never "absent". Consequence for callers: hand *all* parameters
    to the optimizer, since a `requires_grad` filter at construction time
    silently excludes anything unfrozen later.
- `src/neurom/decompositions/pgd_fem_model.py` — **`PGDFEMModel(nn.Module)`**:
  `PGDFEMModel(decomposition: TensorDecomposition, field_layout, loss)`, the
  PGD analogue of `neurom.fem_model.FEMModel`. Registers the decomposition's
  factor fields into the layout at construction
  (`decomposition.register_into(field_layout)`); `forward()` fills the layout
  (`decomposition.fill(field_layout)`) then evaluates the external `loss`
  no-arg callable. Depends only on the `TensorDecomposition` contract, so it
  is format-agnostic (pinned by a fake-decomposition unit test).

### Added — tests

- `tests/unit/decompositions/test_factor.py` — `FactorSpace` connectivity and
  context, and two `MonomSpec` sharing one space.
- `tests/unit/decompositions/test_pgd.py` — `CPPGD`
  construction + freeze state, monom values read back by name from a
  `FieldLayout` after `register_into` + `fill`, `assemble` (single-mode and
  rank-2 sum of outer products), greedy mode management (freeze/activate/zero,
  RuntimeError at max, optimizer growth), and `NeuROMModel` wiring — including
  a format-agnostic test built on a fake `TensorDecomposition`
  (`_ConstantDecomposition`) with no CP structure, to pin that `NeuROMModel`
  depends only on the base-class contract.
- `tests/integration/test_1d_beam_deflection_PGD_test.py` — reference solve:
  1D beam parametrized by Young modulus `E`, a **2-axis** decomposition
  `u(x,E) = Σ_m S_m(x) g_m(E)`, solved through `PGDFEMModel` + `FieldLayout`.
  The parametric energy is defined **in the test**
  (`potential_energy(cppgd, field_layout, f_value)`), assembled monom-by-monom
  by reading `cppgd.separated_view(field_layout)` off the filled layout. Two
  tests: a rank-1 LBFGS solve, and a greedy enrichment solve (mode 0 →
  `add_mode` → mode 1) that stays bounded. Both match the analytical solution
  `u(x,E) = 0.5·f·(x−x_min)(x−x_max)/E` within ~2.6% (tolerance 5%).

### Added — docs

- `docs/superpowers/specs/2026-07-08-cp-pgd-module-design.md` — design spec.
- `docs/superpowers/plans/2026-07-08-cp-pgd-module.md` — task-by-task plan.

### NOT touched (intentionally)

- `tests/integration/test_1d_beam_deflection_PGD.py` — **reserved for the user
  to implement as an exercise.** Do not edit it.
- Pre-existing uncommitted working-tree changes present at session start
  (`.gitignore`, `pyproject.toml`, `tests/integration/test_1d_beam_deflection.py`,
  `uv.lock`) — left as-is; they are the user's.

## Physics / sign convention (beam reference test)

Matches the existing library's `ElasticEnergy - LoadPotential` combination,
which expands to `+½(u')² + f·u`. The parametric energy is

```
J = 0.5 · ∫_E E ∫_x (∂_x u)² dx dE  +  ∫_E ∫_x f u dx dE
```

separated as `elastic = 0.5·Σ_{m,n}[∫∂_xS_m ∂_xS_n dx][∫E g_m g_n dE]` and
`load = Σ_m[∫ f S_m dx][∫ g_m dE]`. Minimizer:
`u(x,E) = 0.5·f·(x−x_min)(x−x_max)/E`.

## How to run

```bash
uv run pytest                                                   # full suite (79)
uv run pytest tests/unit/decompositions/test_pgd.py -v          # module unit tests
uv run pytest tests/integration/test_1d_beam_deflection_PGD_test.py -v
```

(Use `uv run` — bare `python`/`pytest` are not on PATH.)

## Deferred / known Minor items (optional, benign)

- `assemble` with more than 26 axes raises a bare `IndexError`
  (`string.ascii_lowercase[:n_factors]`) rather than a clear message.
- `assemble` rebuilds a `PointWiseInterpolator` per mode (efficiency only;
  it is post-processing and returns a detached tensor).
- `FactorSpace.mapping` is typed `object` (no mapping ABC exists in the codebase yet).

## Reference for the original implementation

The old CP-PGD lives in the sibling checkout
`../neurom_develop_daby/neurom/HiDeNN_PDE.py` (class `NeuROM`), with the
parametric energy in `neurom/src/PDE_Library.py`
(`PotentialEnergyVectorisedParametric`). This module is a modern re-implementation
of that capability on the current `neurom` library.
