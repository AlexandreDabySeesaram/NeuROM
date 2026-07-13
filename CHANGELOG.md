# Changelog

All notable changes to this project are recorded here. Newest entries on top.
Each session that implements something appends an entry. For deep detail on a
change, follow the linked doc.

## 2026-07-13 — NeuROMModel + SeparatedDomain (PGD on a classic nn.Module)

Replaced `PGDFEMModel` / `CPPGD.separated_view` with a decomposition-driven
`NeuROMModel` (train/eval `forward`): training fills the `FieldLayout` and
returns it as the intermediate output an external `energy` consumes
(`out = model(); loss = model.energy(out)`); eval does matched-pointwise
inference. `CPPGD` now fills through a truncation-aware
`SeparatedDomain(IntegrationDomain)` (built once, grows on `add_mode`) and
exposes `directory()` so a separable energy reads modes from the layout by
name. `evaluate`/`assemble` are vector-ready (one vector factor per mode, e.g.
2-D displacement), guarded against >1 vector factor. `TensorDecomposition` ABC
gains `evaluate`/`assemble`; `n_modes_truncated` is now a property delegating to
the domain. Full suite green.
Design: docs/superpowers/specs/2026-07-13-neurom-model-separated-domain-design.md

## 2026-07-10 — CP-PGD on the FieldLayout / FEMModel abstraction

Branch `pgd_addition_solal`. Full suite: 79 passed.

- Added `TensorDecomposition` ABC (`src/neurom/decompositions/base.py`): the
  `register_into(field_layout)` / `fill(field_layout)` contract that a PGD FEM
  model depends on, so CP / future Tucker / TT all drive the same model.
- `CPPGD` now subclasses `TensorDecomposition`:
  - `register_into(layout)` registers all monom fields; `fill(layout)`
    interpolates the active monoms and `update`s them (analogue of
    `IntegrationDomain.interpolate_all`).
  - `separated_view(layout)` reads the active monoms back out of the layout;
    it **replaces** `interpolate_separated()` (removed). `assemble()` unchanged.
- Added `PGDFEMModel(decomposition, field_layout, loss)`
  (`src/neurom/decompositions/pgd_fem_model.py`, exported from
  `neurom.decompositions`): registers factor fields at construction, `forward()`
  fills the layout then evaluates the external loss. Depends only on the ABC —
  a fake decomposition test pins the format-agnosticism.
- No `FieldLayout` changes: CP-PGD uses its existing `add`/`update`/`__getitem__`.
- Migrated the beam integration test and the unit tests onto the layout flow.

Full detail: [design spec](docs/superpowers/specs/2026-07-10-cppgd-fieldlayout-integration-design.md).

## 2026-07-09 — CP-PGD mode management tweaks

Branch `develop_solal`.

- `CPPGD.add_mode()` no longer freezes the currently-active modes on enrichment;
  it only activates (zeros + unfreezes) the new mode and returns its index.
  Freezing is left to the caller so update strategies (e.g. updated/preview
  modes) can be chosen freely.
- `CPPGD.add_mode_to_optimizer(optim, m=None)` now takes an explicit mode index
  (with negative-index support), defaulting to the last-activated mode, and
  raises `IndexError` when out of range.

## 2026-07-08 — CP-PGD module

Branch `develop_solal`, commits `077ca9d`→`5e3cdb4`. Full suite: 69 passed.

- Added `src/neurom/decompositions/` package: `Axis` descriptor + `CPPGD`
  separated-representation model (generalized to `l` axes) with greedy mode
  enrichment. Energy/loss and training loop deliberately kept outside the module.
  - `interpolate_separated()` exposes each monom individually
    (`dict[str, list[QuadratureAssemblyResult]]`) for writing separable energies.
  - `assemble(coords)` returns the full tensor via a dynamic einsum over any
    number of axes.
  - `add_mode()` / `add_mode_to_optimizer()` + freeze/unfreeze for greedy PGD.
- Added `tests/unit/decompositions/test_pgd.py` (construction, separated view,
  assemble incl. rank-2, greedy management).
- Added `tests/integration/test_1d_beam_deflection_PGD_test.py`: parametric 1D
  beam by Young modulus E, external separable energy, LBFGS solve + greedy
  enrichment, matches analytical `0.5·f·(x−x_min)(x−x_max)/E` (~2.6% error).
- Added design spec and implementation plan under `docs/superpowers/`.
- Left `tests/integration/test_1d_beam_deflection_PGD.py` untouched (user exercise).

Full detail: [docs/CP_PGD_IMPLEMENTATION_NOTES.md](docs/CP_PGD_IMPLEMENTATION_NOTES.md).
