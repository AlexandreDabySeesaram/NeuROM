# CP-PGD on the FieldLayout / FEMModel abstraction — design

## Context

The first CP-PGD module (spec `2026-07-08-cp-pgd-module-design.md`) was built as
an **island**: `CPPGD` owns its own meshes / quadrature contexts / monoms and
exposes `interpolate_separated()` and `assemble()`, and the energy is written
entirely outside, monom by monom, directly against that dict. It never touches
the two abstractions the rest of `neurom` is built on:

- **`FieldLayout`** — a *flat* container mapping `name -> QuadratureAssemblyResult`
  (`add(field)` at setup, `update(field, result)` per forward, `layout[name]` to
  read). It has no notion of modes, axes, or products.
- **`FEMModel(mesh, field_layout, integration_domain, loss)`** — orchestrator
  whose `forward()` is `integration_domain.interpolate_all(field_layout)` then
  `loss()`.

Goal of this change: make CP-PGD **fill the `FieldLayout` it is given** and be
driven through a PGD analogue of `FEMModel`, so it stops being a parallel island
and lives inside the library's dependency-injected pipeline. A second, forward
looking goal: the PGD FEM model must **not depend on the CP format** so that
future `TuckerPGD` and `TTPGD` decompositions — which have different structures
and different numbers of factor fields — drop into the exact same model and
layout.

```
CP    u = sum_m           prod_k w_m^k(x_k)
Tucker u = sum_{j1..jd} G_{j1..jd} prod_k X^k_{jk}(x_k)
TT    u = sum_{a1..a{d-1}} X^1_{a1}(x1) X^2_{a1 a2}(x2) ... X^d_{a{d-1}}(xd)
```

## Decisions (from brainstorming)

1. **Structure lives in the decomposition; the layout stays flat.** `CPPGD`
   fills the flat `FieldLayout` with each monom's interpolation, keyed by the
   monom field name. The separable energy recovers the mode/axis structure from
   the decomposition object, not from the layout.
2. **The energy stays an external function.** No `Term`/`PhysicsLoss` algebra for
   the separated energy (it is inherently multi-domain, double-summed over mode
   pairs, and problem-specific — e.g. it must know the `E` axis carries the
   stiffness). It is refactored only to read from the filled layout.
3. **A new PGD variant model**, `PGDFEMModel`, is added; the existing `FEMModel`
   is left untouched.
4. **The layout-based flow becomes canonical.** `interpolate_separated()` is
   dropped in favour of `fill` + `separated_view`; existing PGD tests are
   migrated onto the new flow. `assemble()` is kept for post-processing.
5. **`PGDFEMModel` depends on an abstraction, not on CP.** A new
   `TensorDecomposition` ABC declares the only two methods the model needs
   (`register_into`, `fill`). `CPPGD` implements it; `TuckerPGD` / `TTPGD` will
   too.

## Goals

- `CPPGD` registers its factor fields into, and fills, a caller-supplied
  `FieldLayout`, using the existing `FieldLayout` contract **unchanged**.
- A `PGDFEMModel(decomposition, field_layout, loss)` that is **format-agnostic**:
  it depends only on the `TensorDecomposition` ABC.
- The 1D parametric beam solves end-to-end through
  `PGDFEMModel` + `FieldLayout` + an external separable energy, matching the same
  analytical solution as before.

## Non-goals

- **No `FieldLayout` changes.** The whole point is that CP-PGD works with the
  abstraction as it stands (`add` / `update` / `__getitem__`).
- **No generic/reusable separable `Term` layer.** The energy remains an external
  function (decision 2). Folding it into an enriched `physics` layer stays future
  work.
- **No Tucker / TT implementation in this session.** Only the ABC seam that makes
  them possible later. The ABC is validated by a trivial fake decomposition in
  the tests, not by a real second format.
- No change to greedy enrichment semantics, `assemble`, or the `Axis` descriptor.

## Architecture

### The seam: `TensorDecomposition` ABC — `src/neurom/decompositions/base.py`

```python
class TensorDecomposition(nn.Module, ABC):
    """A tensor/separated decomposition that can populate a FieldLayout.

    Concrete formats (CP, and later Tucker / TT) own their own factor fields and
    know how many they are. The only contract PGDFEMModel relies on is: register
    those fields once, then re-interpolate ('fill') the active ones per forward.
    Format-specific structure readback, assembly and rank/mode enrichment stay on
    the concrete subclass.
    """

    @abstractmethod
    def register_into(self, field_layout) -> None:
        """Register this decomposition's factor fields in the layout (setup)."""

    @abstractmethod
    def fill(self, field_layout) -> None:
        """Interpolate the active factor fields and update() them in the layout
        (called once per forward, the PGD analogue of
        IntegrationDomain.interpolate_all)."""
```

`nn.Module + ABC` follows the existing codebase idiom (`FieldBase(nn.Module,
ABC)`). Exactly these two methods — no more — so the model dependency is minimal.
`register_into` / `fill` never count fields: each concrete format registers and
fills however many factor fields it owns (CP: `axes x modes` monoms; Tucker: `d`
factor fields plus an internally-held core tensor `G`; TT: `d` rank-indexed
cores). That is what makes the model independent of the number of fields.

### `CPPGD(TensorDecomposition)` — `src/neurom/decompositions/pgd.py`

Constructor, `Axis`, monom grid, per-axis `Mesh`/`QuadratureContext`, and all
mode management (`add_mode`, `freeze_mode`/`unfreeze_mode`, `_zero_out`,
`add_mode_to_optimizer`) are **unchanged**. Monom field names remain
`f"{axis.name}_mode{m}"` (e.g. `"space_mode0"`) — these become the flat layout
keys.

Changed / added:

- **`register_into(field_layout)`** — registers *all* `n_modes_max x len(axes)`
  monom `TrainableField`s via `field_layout.add(monom)`. Called once at setup.
  Registering all modes (including not-yet-active ones) keeps `add_mode` free of
  any layout dependency (unchanged from today). Inactive monoms are registered
  but never interpolated, so `layout[name]` for an inactive monom correctly
  raises `RuntimeError` ("registered but not yet interpolated") — but nothing in
  the active flow reads them.

- **`fill(field_layout)`** — for each **active** mode `m` (`m in
  range(n_modes_truncated)`) and each axis `k`, run
  `QuadratureAssembly(self._contexts[k], axis.sf, self.monoms[m][k]).interpolate()`
  and `field_layout.update(self.monoms[m][k], result)`. This is the CP analogue
  of `IntegrationDomain.interpolate_all`.

- **`separated_view(field_layout) -> dict[str, list[QuadratureAssemblyResult]]`**
  — for each axis, a list indexed by active mode, each entry read back from the
  layout: `field_layout[self.monoms[m][k].name]`. Returns the same shape the old
  `interpolate_separated()` returned; this is what the external energy consumes.
  Requires the layout to have been `fill`ed first (guaranteed inside
  `PGDFEMModel.forward`, which fills before calling the loss).

- **Drop `interpolate_separated()`** — superseded by `fill` + `separated_view`.

- **Keep `assemble(coords)`** unchanged (post-processing / viz / tests).

**Parameter double-reference note.** After `register_into`, each monom is
reachable both through `cppgd.monoms` and through `field_layout._fields`. This is
the *same* pattern the standard pipeline already uses (a `TrainableField` lives
in both the `FieldLayout` and the `QuadratureAssembly.field`). `nn.Module`
parameter iteration deduplicates by identity, so `PGDFEMModel.parameters()`
yields each monom parameter once and optimizer construction is unaffected.

### `PGDFEMModel(nn.Module)` — `src/neurom/decompositions/pgd_fem_model.py`

```python
class PGDFEMModel(nn.Module):
    def __init__(self, decomposition: TensorDecomposition, field_layout, loss):
        super().__init__()
        self.decomposition = decomposition
        self.field_layout = field_layout
        self.loss = loss
        decomposition.register_into(field_layout)   # register factor fields once

    def forward(self):
        self.decomposition.fill(self.field_layout)   # analogue of interpolate_all
        return self.loss()
```

- Depends **only** on the `TensorDecomposition` ABC — CP is invisible to it, so
  `TuckerPGD` / `TTPGD` slot in unchanged.
- Lives in `decompositions/` (not top-level `fem_model.py`) so the dependency
  direction stays one-way: `decompositions -> core`, never core -> decompositions.
- `loss` is a no-arg callable, called as `self.loss()` — exactly how `FEMModel`
  invokes its `loss`. The problem author closes it over `(decomposition,
  field_layout, problem params)`.
- Exported from `neurom.decompositions` alongside `Axis`, `CPPGD`,
  `TensorDecomposition`.

## Data flow (beam example, energy lives in the test)

Setup (once):

```python
field_layout = FieldLayout()                 # flat, "structureless"
cppgd        = CPPGD(axes=[space_axis, E_axis], n_modes_max=..., n_modes_ini=1)
loss         = lambda: potential_energy(cppgd, field_layout, f_value)
model        = PGDFEMModel(cppgd, field_layout, loss)   # registers monoms
```

Per optimizer step:

```python
def closure():
    optimizer.zero_grad()
    l = model()            # fill(layout) -> loss(): separated_view(layout) -> J
    l.backward(retain_graph=True)
    return l
```

Energy (external, unchanged maths, new source):

```python
def potential_energy(cppgd, field_layout, f_value):
    sep   = cppgd.separated_view(field_layout)   # was cppgd.interpolate_separated()
    space, para = sep["space"], sep["E"]
    ...  # identical double-sum elastic + load assembly as today
```

Greedy enrichment is still driven by the training loop: `cppgd.freeze_mode(m)`,
`cppgd.add_mode()`, rebuild the optimizer (or `add_mode_to_optimizer`). None of
that touches `PGDFEMModel`.

## Testing

Unit — `tests/unit/decompositions/test_pgd.py` (migrated):

- Replace every `interpolate_separated()` assertion with the `FieldLayout` flow:
  `register_into(layout)` then `fill(layout)` then `separated_view(layout)`, and
  assert the same keys / shapes / values as before.
- `register_into` populates the layout with all `n_modes_max x axes` monom names;
  reading an inactive monom via `layout[name]` raises `RuntimeError`.
- `fill` only updates active modes; after `add_mode` + `fill` the new mode
  appears in `separated_view`.
- A `PGDFEMModel` wiring test: `model()` returns a scalar; one optimizer step
  runs and changes parameters.
- A **format-agnostic** test: a trivial fake `TensorDecomposition` (registers one
  `Field`, fills it with a constant result) drives `PGDFEMModel` with a dummy
  loss — proving the model has no CP dependency.

Integration — `tests/integration/test_1d_beam_deflection_PGD_test.py` (migrated):

- Both tests (rank-1 LBFGS solve, and greedy mode-2 enrichment) now build a
  `FieldLayout`, a `CPPGD`, and a `PGDFEMModel`, optimize via `model()`, and check
  the assembled `u(x, E)` against `0.5 f (x - x_min)(x - x_max) / E` within the
  existing tolerance. Energy read via `separated_view`.

Full suite must stay green (`uv run pytest`).

## Docs

- CHANGELOG entry (newest on top).
- Refresh the "Files" / API section of `docs/CP_PGD_IMPLEMENTATION_NOTES.md` to
  describe `register_into` / `fill` / `separated_view` / `PGDFEMModel` /
  `TensorDecomposition` and note the removal of `interpolate_separated()`.
- This spec, plus the implementation plan under `docs/superpowers/plans/`.

## Open questions / future work

- `TuckerPGD` / `TTPGD` concrete decompositions on the same `TensorDecomposition`
  seam (core tensor `G` as internal trainable state read by the energy; TT cores
  as rank-indexed factor fields).
- Folding the separable energy into an enriched `physics` layer once the forms
  stabilize across formats.
- Relevement (lifting) for non-homogeneous BCs; r-adaptivity of axis meshes.
