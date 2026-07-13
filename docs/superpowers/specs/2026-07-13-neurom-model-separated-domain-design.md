# NeuROMModel + SeparatedDomain: PGD on a classic-nn-Module model — design spec

**Date:** 2026-07-13
**Branch:** `pgd_addition_solal`
**Supersedes:** the `PGDFEMModel` + `CPPGD.separated_view` integration
(`docs/superpowers/specs/2026-07-10-cppgd-fieldlayout-integration-design.md`).

## Context

The 2026-07-10 work made `CPPGD` fill a `FieldLayout` and drive a `PGDFEMModel`.
On review, two things did not sit right with the library's philosophy:

1. `PGDFEMModel.forward` returned the scalar energy directly (FEMModel-style),
   and the separable energy read the modes through a bespoke
   `CPPGD.separated_view` that returned live `QuadratureAssemblyResult`s — a side
   channel that bypasses the `FieldLayout` name-lookup contract every existing
   `Term` uses.
2. `CPPGD.fill` rebuilt a `QuadratureAssembly` for every monom on every forward,
   instead of going through the `IntegrationDomain` abstraction the rest of the
   library uses to cache assemblies and dedup contexts.

This spec replaces `PGDFEMModel` with a `NeuROMModel` that reads like a classic
`nn.Module` (a train/eval `forward`), routes the separable energy through the
standard `field_layout[name]` lookup via a new `CPPGD.directory()`, and makes
`fill` go through a new truncation-aware `SeparatedDomain(IntegrationDomain)`.

## Goals

- `NeuROMModel`: an `nn.Module` whose `forward` branches on `self.training` —
  training produces the intermediate output an external energy consumes; eval is
  pointwise inference. Format-agnostic (depends only on `TensorDecomposition`).
- `CPPGD.fill` goes through `SeparatedDomain`, which builds one assembly per
  monom **once** and interpolates only the **active** modes.
- The separable energy reads modes through the layout by name, via
  `CPPGD.directory()` — no `separated_view`, no `QuadratureAssemblyResult` side
  channel.

## Non-goals

- No reusable library `Term` for the separable PGD energy yet. The energy stays
  a **test-local function** (it "will be precised later"). This spec only fixes
  the plumbing (`directory()` → `field_layout[name]`) that such a `Term` will
  later use.
- No Tucker/TT implementation. The `TensorDecomposition` ABC is extended only
  where `NeuROMModel` needs it; `directory()` stays CP-specific until a second
  format needs a shared shape.

## Design decisions (resolved forks)

| Fork | Decision |
|---|---|
| Training `forward` output | Returns the **intermediate output** (the filled `field_layout`); energy is applied **outside**: `output = model(); loss = model.energy(output)`. |
| Active-mode interpolation | New **`SeparatedDomain(IntegrationDomain)`**: assemblies grouped into mode-blocks, built once, interpolates only active blocks, `grow()` on `add_mode`. |
| Energy location | **Test-local function**, rewritten to read the layout via `directory()`. No new library `Term`. |
| Eval `forward` | `forward(coords)` → **matched pointwise** (`PointWiseInterpolator`, diagonal). `assemble()` stays for the full outer-product grid. |
| `directory()` shape | **Axis-major, active modes only**: `dict[axis_name -> list[monom_name]]`, ordered by mode, truncated to active modes. |

## Components

### 1. `SeparatedDomain(IntegrationDomain)` — new file `src/neurom/interpolation/separated_domain.py`

A truncation-aware `IntegrationDomain`. Assemblies are grouped into **mode-blocks**
(one block per mode = one `QuadratureAssembly` per axis). Only the first
`n_active_modes` blocks are interpolated, matching greedy PGD truncation.

```python
class SeparatedDomain(IntegrationDomain):
    """IntegrationDomain over mode-blocked assemblies; interpolates only active modes.

    Assemblies are grouped into blocks (one block per mode, each block one
    assembly per axis). `interpolate_all` interpolates only the first
    `n_active_modes` blocks and `update()`s them in the layout. `grow()`
    activates the next block (greedy PGD enrichment).
    """

    def __init__(self, mode_blocks: list[list[QuadratureAssembly]], n_active_modes: int):
        flat = [a for block in mode_blocks for a in block]
        super().__init__(flat)                     # dedups contexts, registers assemblies
        self._mode_blocks = mode_blocks            # plain list-of-lists; same objects as self.assemblies
        self.register_buffer("n_active_modes", torch.tensor(int(n_active_modes)))

    def grow(self) -> int:
        """Activate the next mode-block. Returns its index. Raises at capacity."""
        if int(self.n_active_modes) >= len(self._mode_blocks):
            raise RuntimeError("Cannot grow: all mode-blocks already active.")
        idx = int(self.n_active_modes)
        self.n_active_modes += 1
        return idx

    def interpolate_all(self, field_layout):
        for block in self._mode_blocks[: int(self.n_active_modes)]:
            for assembly in block:
                field_layout.update(assembly.field, assembly.interpolate())
```

**Notes / invariants**
- `n_active_modes` is a **registered buffer** → it round-trips through
  `state_dict`/`load_state_dict` (checkpointing is a model responsibility).
- `_mode_blocks` holds the *same* `QuadratureAssembly` objects already registered
  in the parent's `self.assemblies` `ModuleList`; the plain list does not
  re-register them, and `nn.Module` dedups parameters by identity.
- `update_contexts()` (inherited) still recomputes geometry for all unique
  contexts — used only if axis nodes become trainable (not in the beam test).
- Export `SeparatedDomain` from `src/neurom/interpolation/__init__.py`.

### 2. `CPPGD` changes — `src/neurom/decompositions/pgd.py`

**New constructor argument `name: str`.** Monoms are named
`f"{self.name}_dim{a.name}_mode{m}"` (per the review's spec; e.g. name `"beam"`,
axis `"space"`, mode 0 → `"beam_dimspace_mode0"`). This makes monom field names
globally unique so several decompositions can share one `FieldLayout` later.

**`SeparatedDomain` ownership + single source of truth for active modes.**
At construction, after building meshes, contexts, and monoms, build one
`QuadratureAssembly(self._contexts[k], axis.sf, self.monoms[m][k])` per monom,
group them into mode-blocks, and construct
`self.domain = SeparatedDomain(mode_blocks, n_active_modes=min(n_modes_ini, n_modes_max))`.

`n_modes_truncated` becomes a **read-only property** delegating to the domain
(single source of truth, avoids two out-of-sync buffers):

```python
@property
def n_modes_truncated(self) -> int:
    return int(self.domain.n_active_modes)
```

(The old `register_buffer("n_modes_truncated", ...)` is removed. All existing
`int(self.n_modes_truncated)` reads keep working; the property is int-valued.
`state_dict` still persists the count — now under `domain.n_active_modes`.)

**Construction order (important):** because `n_modes_truncated` now reads
`self.domain`, `self.domain` must be assigned **before** the constructor's
closing `freeze_all()` / `unfreeze_mode(m)` loop (which reads
`self.n_modes_truncated`). Order: meshes → contexts → monoms → mode-block
assemblies → `self.domain` → freeze/unfreeze.

**Method changes:**
- `fill(field_layout)` → delegates: `self.domain.interpolate_all(field_layout)`.
- `add_mode()` → `idx = self.domain.grow()`, then `self._zero_out(idx)`,
  `self.unfreeze_mode(idx)`, return `idx`. (Freeze state of active modes untouched,
  as today — caller's job.) Raises when the domain is at capacity.
- `register_into(field_layout)` — unchanged (registers all monoms, all modes).
- **Remove** `separated_view()`.
- **Add** `directory() -> dict[str, list[str]]` (axis-major, active modes only):
  ```python
  def directory(self) -> dict[str, list[str]]:
      n = self.n_modes_truncated
      return {
          axis.name: [self.monoms[m][k].name for m in range(n)]
          for k, axis in enumerate(self.axes)
      }
  ```
- **Add** `evaluate(coords) -> torch.Tensor` — matched pointwise (diagonal),
  the `TensorDecomposition` inference contract. **Vector-ready**: each factor
  keeps its own field dim and factors are combined by broadcasting, so a mode's
  single vector factor (e.g. a 2-D displacement `S_m(x)`) multiplied by scalar
  weights `g_m(μ)` yields the vector field with no special-casing:
  ```python
  def evaluate(self, coords):
      """u at matched points. coords: one 1-D tensor per axis, all length P.
      Returns (P, d) = sum_m prod_k w_m^k(coords[k][p]); d is the (single)
      vector factor's dim, or 1 if every factor is scalar. Detached."""
      n = self.n_modes_truncated
      total = None
      for m in range(n):
          prod = None
          for k, axis in enumerate(self.axes):
              pwi = PointWiseInterpolator(self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping)
              w = pwi.at_position(coords[k].reshape(-1))   # (P, 1, dim_k)
              w = w.reshape(w.shape[0], -1)                # (P, dim_k)
              prod = w if prod is None else prod * w       # scalar * vector broadcasts
          total = prod if total is None else total + prod
      return total                                          # (P, d)
  ```
  All `coords[k]` must share length `P`. Shape trace (2-D displacement, param
  scalar): space `(P, 2)` · param `(P, 1)` → `(P, 2)`.
- **Change** `assemble(coords)` — **vector-ready** full grid tensor. A vector
  factor turns that axis's per-axis array into `(n_modes, N_k, d_k)` and appends
  one component index to the einsum output; scalar-only collapses to today's
  `(N_1, ..., N_l)`:
  ```python
  def assemble(self, coords):
      """Full grid tensor, shape (N_1, ..., N_l[, d]). The trailing d is present
      iff a vector factor exists (else dropped). Detached."""
      n_modes = self.n_modes_truncated
      mode_letter = "Z"
      per_axis = []                                          # (n_modes, N_k) or (n_modes, N_k, d_k)
      for k, axis in enumerate(self.axes):
          P_k = coords[k].reshape(-1).shape[0]
          cols = []
          for m in range(n_modes):
              pwi = PointWiseInterpolator(self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping)
              w = pwi.at_position(coords[k].reshape(-1)).reshape(P_k, -1)   # (N_k, d_k)
              cols.append(w.reshape(-1) if w.shape[1] == 1 else w)         # (N_k,) or (N_k, d_k)
          per_axis.append(torch.stack(cols, dim=0))

      grid_letters = string.ascii_lowercase[: len(self.axes)]              # N-dims, in output
      comp_pool = iter(c for c in string.ascii_uppercase if c != mode_letter)
      in_subs, out_grid, out_comp = [], "", ""
      for k, arr in enumerate(per_axis):
          sub = mode_letter + grid_letters[k]
          out_grid += grid_letters[k]
          if arr.dim() == 3:                                               # vector axis
              c = next(comp_pool)
              sub += c
              out_comp += c
          in_subs.append(sub)
      return torch.einsum(f"{','.join(in_subs)}->{out_grid}{out_comp}", *per_axis)
  ```
  Grid letters are lowercase (N-dims), the summed mode index is `Z`, component
  letters are uppercase (≠ `Z`) — no subscript collision. Output layout: grid
  dims first, then the component dim.

**Vector-factor precondition (guarded at construction).** CP admits **at most
one vector-valued factor per mode** — the physical/displacement field; the rest
are scalar weights. (Two vector factors make `Π_k w_m^k` ill-defined: `evaluate`
would broadcast-error, `assemble` would silently form a component outer product.)
The constructor validates this once, since all monoms on an axis share
`init_values` dim:
```python
if sum(int(a.init_values.shape[1] > 1) for a in self.axes) > 1:
    raise ValueError("CP-PGD admits at most one vector-valued factor per mode.")
```

### 3. `NeuROMModel(nn.Module)` — new file `src/neurom/neurom_model.py`

Top-level (sibling of `fem_model.py`); the decomposition-driven counterpart of
`FEMModel`. Format-agnostic — depends only on the `TensorDecomposition` contract.

```python
class NeuROMModel(nn.Module):
    def __init__(self, field_layout, decomposition: TensorDecomposition, energy):
        super().__init__()
        self.field_layout = field_layout
        self.decomposition = decomposition
        self.energy = energy                       # injected callable(output) -> scalar
        decomposition.register_into(field_layout)  # register factor fields once (pass a fresh layout)

    def forward(self, coords=None):
        if self.training:
            self.decomposition.fill(self.field_layout)
            return self.field_layout               # intermediate output; energy applied outside
        if coords is None:
            raise ValueError("eval forward requires coords (per-axis, matched length).")
        return self.decomposition.evaluate(coords)

    def assemble(self, coords):
        return self.decomposition.assemble(coords)
```

**Notes**
- `energy` is a plain injected callable (DI, like `FEMModel.loss`). Training use:
  `output = model(); loss = model.energy(output)`. If a future `PhysicsLoss`-based
  energy is passed, being an `nn.Module` it registers as a submodule harmlessly.
- `decomposition` and `field_layout` are submodules → `model.parameters()`
  surfaces the monom parameters; the monoms are shared (by identity) between
  `decomposition`, `field_layout`, and the domain's assemblies, so `nn.Module`
  dedups them — no double counting.
- Same **fresh-layout** contract as before: `__init__` registers into the layout,
  so a layout already holding those names raises `ValueError`.

### 4. `TensorDecomposition` ABC — `src/neurom/decompositions/base.py`

Extend so `NeuROMModel` is fully format-agnostic in both modes:

```python
@abstractmethod
def register_into(self, field_layout) -> None: ...
@abstractmethod
def fill(self, field_layout) -> None: ...
@abstractmethod
def evaluate(self, coords): ...      # matched pointwise inference
@abstractmethod
def assemble(self, coords): ...      # full grid tensor
```

`directory()` stays **CP-specific** (not on the ABC): Tucker's core-tensor
structure won't share this shape, so YAGNI until a second format needs it.

### 5. File moves & exports

- **Delete** `src/neurom/decompositions/pgd_fem_model.py`.
- **Add** `src/neurom/neurom_model.py` (`NeuROMModel`).
- **Add** `src/neurom/interpolation/separated_domain.py` (`SeparatedDomain`);
  export it from `src/neurom/interpolation/__init__.py`.
- `src/neurom/decompositions/__init__.py`: drop `PGDFEMModel`; keep
  `TensorDecomposition`, `Axis`, `CPPGD`.
- `NeuROMModel` is imported as `from neurom.neurom_model import NeuROMModel`
  (top-level `neurom/__init__.py` only exposes `__version__`; the library uses
  full-path imports, matching `from neurom.fem_model import FEMModel`).

## Energy rewrite (test-local)

`potential_energy` reads the modes through the layout by name, via `directory()`:

```python
def potential_energy(cppgd, field_layout, f_value):
    d = cppgd.directory()                              # {'space': [...], 'E': [...]}
    space = [field_layout[name] for name in d["space"]]  # QuadratureAssemblyResult per mode
    para  = [field_layout[name] for name in d["E"]]
    n_modes = len(space)
    # ... identical separable elastic + load math as before, reading .x/.u/.measure ...
```

The math (`elastic = 0.5·Σ_{m,n} Kx·AE`, `load = Σ_m Fx·Gm`) and the analytical
target `u(x,E) = 0.5·f·(x−x_min)(x−x_max)/E` are unchanged; only the source of
the per-mode `QuadratureAssemblyResult`s changes (layout-by-name instead of
`separated_view`).

## Testing strategy

**`tests/integration/test_1d_beam_deflection_PGD_test.py`**
- `CPPGD(name="beam", axes=[...], n_modes_max=..., n_modes_ini=...)`.
- Wire through `NeuROMModel(field_layout, cppgd, energy=lambda out: potential_energy(cppgd, out, f_value))`.
- Training loop: `output = model(); loss = model.energy(output); loss.backward(...)`.
- Final check: `model.assemble([x_test, E_test])` vs analytical (≤5%), unchanged.
- Add an **eval-mode** assertion: `model.eval(); model([x_pts, E_pts])` (matched
  pointwise) agrees with the analytical field at the same matched points.
- Greedy test: `cppgd.freeze_mode(0); cppgd.add_mode()` (now also grows the
  domain) then re-solve; rank-2 field still matches the rank-1 analytical target.

**`tests/unit/decompositions/test_pgd.py`**
- Remove `separated_view` / `interpolate_separated` tests.
- `SeparatedDomain`: only active blocks interpolated (inactive monom → layout
  read raises `RuntimeError`); `grow()` activates the next block and raises at
  capacity; contexts deduped.
- `directory()`: axis-major keys, one name per active mode, names match
  `f"{name}_dim{axis}_mode{m}"`, grows after `add_mode`.
- `evaluate(coords)`: matched pointwise equals the analytical/`assemble` diagonal.
- **Vector factor** (`dim > 1`): a 2-axis CP with a 2-D vector factor on one axis
  and a scalar weight on the other — `evaluate` returns `(P, 2)` and `assemble`
  returns `(N_0, N_1, 2)`, both matching a hand-computed `Σ_m S_m ⊗ g_m`. Also:
  constructing with **two** vector axes raises `ValueError`.
- `NeuROMModel`: train `forward()` returns the filled `field_layout` and
  optimizes; eval `forward(coords)` returns pointwise values; `assemble`
  delegates. Format-agnostic test: a fake `TensorDecomposition` implementing all
  four abstract methods (`register_into`/`fill`/`evaluate`/`assemble`) drives
  `NeuROMModel` with no CP structure.
- `n_modes_truncated` property tracks `domain.n_active_modes` through `add_mode`.

## Edge cases

- `evaluate` with mismatched `coords` lengths — matched semantics require equal
  `P` per axis; document the precondition (assemble handles the grid case).
- `add_mode` at capacity — `domain.grow()` raises `RuntimeError`; `CPPGD.add_mode`
  propagates it.
- Passing a non-fresh `field_layout` to `NeuROMModel` — `register_into` hits the
  `FieldLayout.add` duplicate-name `ValueError`.
- More than one vector-valued axis — constructor raises `ValueError` (CP admits
  at most one vector factor per mode).

## Out of scope / deferred

- Reusable `SeparablePGDEnergy` `Term` (energy stays test-local).
- Tucker/TT decompositions (the ABC seam is ready; `directory()` shape is not
  yet generalized).
- More than one vector factor per mode (mathematically ill-defined in CP; guarded
  at construction). Single-vector-factor modes ARE supported by `evaluate`/`assemble`.
