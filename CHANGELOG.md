# Changelog

All notable changes to this project are recorded here. Newest entries on top.
Each session that implements something appends an entry. For deep detail on a
change, follow the linked doc.

## 2026-07-10 — TensorDecomposition ABC + CPPGD.register_into/fill

Branch `pgd_addition_solal`, commit `1208881`. Full suite: 75 passed.

- Added `src/neurom/decompositions/base.py`: `TensorDecomposition` ABC
  (`nn.Module` + `ABC`) with abstract `register_into(field_layout)` and
  `fill(field_layout)` — the seam a future `PGDFEMModel` will depend on.
- `CPPGD` now subclasses `TensorDecomposition` and implements both methods:
  - `register_into` registers every monom (all `n_modes_max` modes x all
    axes, including not-yet-active ones) in a `FieldLayout` up front.
  - `fill` re-interpolates only the currently-active monoms per axis and
    `update()`s them in the layout (CP analogue of
    `IntegrationDomain.interpolate_all`). Inactive monoms stay registered but
    uninterpolated (reading them raises `RuntimeError`, per `FieldLayout`
    contract).
- Exported `TensorDecomposition` from `neurom.decompositions`.
- `src/neurom/field_layout.py` untouched — `CPPGD` only uses its existing
  `add`/`update`/`__getitem__` contract.
- Extended `tests/unit/decompositions/test_pgd.py` with 4 tests covering the
  ABC relationship, registration completeness, fill correctness (matches
  direct `QuadratureAssembly`), and inactive-monom non-interpolation.

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
