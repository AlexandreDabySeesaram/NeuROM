# Changelog

All notable changes to this project are recorded here. Newest entries on top.
Each session that implements something appends an entry. For deep detail on a
change, follow the linked doc.

## 2026-07-20 — NeuROMModel: `energy` → `loss`; optimizer wiring moved off the PGD (BREAKING)

- `NeuROMModel.__init__` renames the injected callable `energy` → `loss` (and
  the `model.energy(...)` accessor → `model.loss(...)`), matching `FEMModel.loss`.
- `CPPGD.add_mode_to_optimizer` moved to `NeuROMModel.add_mode_to_optimizer`, so
  the decomposition stays agnostic to the optimizer. `CPPGD` now only reports a
  mode's trainable tensors via the new `CPPGD.mode_parameters(m=None)`; the model
  does the `optim.add_param_group` wiring. Callers switch from
  `pgd.add_mode_to_optimizer(optim)` to `model.add_mode_to_optimizer(optim)`.

## 2026-07-16 — one IntegrationDomain per problem; SeparatedDomain removed (BREAKING)

One `IntegrationDomain` now interpolates every field of a problem — the PGD
monoms **and** other fields such as loads — fixing the case where a load added
to the `FieldLayout` was never interpolated (`Field 'load' registered but not
yet interpolated`). Changes:

- `QuadratureAssembly` gains an `active` bool buffer (default `True`) + `activate()`;
  `IntegrationDomain.interpolate_all` skips inactive assemblies. Mode truncation
  lives here now, per assembly.
- `SeparatedDomain` is **deleted**.
- `Axis` builds and exposes its own `.mesh` / `.context` (`__post_init__`); the
  mapping stays injected. Non-monom fields share `axis.context`.
- `CPPGD` no longer owns a domain or builds meshes: it reads `axis.context`,
  owns mode-blocked flagged assemblies, derives `n_modes_truncated` from the
  flags, and exposes `assemblies()`. `fill()` removed.
- `TensorDecomposition` drops `fill()` from its contract (`register_into` /
  `evaluate` / `assemble` remain).
- `NeuROMModel(field_layout, decomposition, integration_domain, energy)` now
  takes an injected domain (mirroring `FEMModel`); its training forward runs
  `integration_domain.interpolate_all`.

Design: `docs/superpowers/specs/2026-07-16-single-integration-domain-design.md`.
Plan: `docs/superpowers/plans/2026-07-16-single-integration-domain.md`.

## 2026-07-15 — add_mode() no longer zeroes the new mode

`CPPGD.add_mode()` now activates + unfreezes the new mode **without** zeroing its
monoms (removed `_zero_out`); the mode keeps its `Axis.init_values` seed. An
all-zero mode is a stationary point of the energy — every gradient component is
proportional to the other factor, so both factors stay locked at 0 and the mode
never takes off under a gradient optimizer. A non-zero (parametric) seed lets the
linear load term drive the greedy enrichment. Updated the unit test that pinned
the old zero-out contract.

## 2026-07-15 — evaluate() takes a (P, n_axes) point tensor (BREAKING)

`TensorDecomposition.evaluate` / `NeuROMModel.forward` (eval mode) now take the
query points as a single `(P, n_axes)` tensor, **one point per row**
(`pts[p] == (x_p, E_p, ...)`), instead of a list of one 1-D tensor per axis. This
is the more natural human-facing form; `CPPGD._as_axis_columns` unbinds it into
the internal per-axis columns and validates the shape (2-D, column count ==
n_axes) with a clear error. `assemble()` is unchanged — it still takes the
per-axis list (independent lengths, tensor-product grid). Updated all call sites
in the unit/integration tests (`torch.stack([x, E], dim=1)`).

## 2026-07-13 — NeuROMModel + SeparatedDomain (PGD on a classic nn.Module)

Branch `pgd_addition_solal`. Full suite: 93 passed. Feature commits
`6e41670..1097222`.

Replaced `PGDFEMModel` / `CPPGD.separated_view` with a decomposition-driven
`NeuROMModel` (top-level `src/neurom/neurom_model.py`), the counterpart of
`FEMModel` for separated representations, whose `forward` branches on
`self.training`:

- **Training:** fills the `FieldLayout` via the decomposition and returns it as
  the intermediate output an external `energy` consumes
  (`out = model(); loss = model.energy(out)`) — energy is an injected callable,
  like `FEMModel.loss`.
- **Inference:** `forward(coords)` returns the matched-pointwise field; `assemble`
  delegates to the decomposition for the full grid.

Supporting changes:

- Added `SeparatedDomain(IntegrationDomain)`
  (`src/neurom/interpolation/separated_domain.py`): assemblies grouped into
  per-mode blocks, built once, interpolates only the active modes, `grow()` on
  greedy enrichment. `CPPGD.fill` now delegates to it (no more per-forward
  `QuadratureAssembly` rebuild).
- `CPPGD` gains a `name` (monoms named `f"{name}_dim{axis.name}_mode{m}"`) and
  `directory()` (axis-major, active-mode monom field names) so a separable
  energy reads modes from the layout **by name** — no `separated_view` side
  channel. `n_modes_truncated` is now a property delegating to the domain
  (single source of truth for the active-mode count).
- `evaluate`/`assemble` are **vector-ready**: one vector factor per mode (e.g. a
  2-D displacement `S_m(x)` times scalar weights), guarded by a constructor
  `ValueError` against more than one vector factor. Scalar output shape is
  unchanged.
- `TensorDecomposition` ABC gains abstract `evaluate`/`assemble`; `NeuROMModel`
  depends only on this contract (pinned by a fake-decomposition test), so
  future Tucker / TT formats drive the same model.
- Migrated the parametric-beam integration test onto `NeuROMModel` + `directory`
  (energy physically unchanged) and removed `PGDFEMModel` + `separated_view`.

Design: docs/superpowers/specs/2026-07-13-neurom-model-separated-domain-design.md
Plan: docs/superpowers/plans/2026-07-13-neurom-model-separated-domain.md

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
