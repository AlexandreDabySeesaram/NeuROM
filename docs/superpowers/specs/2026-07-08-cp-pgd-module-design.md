# CP-PGD module — design

## Context

NeuROM is being extended with a priori tensor-decomposition (Proper Generalized
Decomposition, PGD). The solution is sought in canonical-polyadic (CP) separated
form over `l` arbitrary axes:

```
u({x_k}_{k=1..l}) = sum_{m=1..M}  prod_{k=1..l}  w_m^k(x_k)
```

We call `w_m^k` a **monom** and `u_m = prod_k w_m^k` a **mode** (following
CLAUDE.md). Modes are obtained by global minimization of the reformulated FEM
energy (Daby-Seesaram et al. 2026, part II), enriched greedily one mode at a
time.

The old reference implementation is the `NeuROM` class in
`neurom_develop_daby/neurom/HiDeNN_PDE.py`. This spec re-implements the same
CP-PGD capability on top of the **current** modular `neurom` library
(`Field`/`TrainableField`, `QuadratureContext`, `QuadratureAssembly`,
`PointWiseInterpolator`, `Term`/`PhysicsLoss`).

First application: `tests/integration/test_1d_beam_deflection_PGD.py` — a 1D beam
parametrized by its Young modulus `E`, so a 2-axis decomposition
`u(x, E) = sum_m S_m(x) g_m(E)`. But the 2-axis case is only a special case; the
module must generalize to any number of axes.

## Goals

- A pure **separated-representation model** for CP-PGD, generalized to `l` axes.
- Faithful port of the `NeuROM` **greedy mode-enrichment** machinery
  (add one mode, freeze the previous ones, unfreeze the new one).
- Two evaluation views: full assembled tensor, and per-axis / per-mode separated
  components.

## Non-goals

- The **energy / loss lives outside the module.** It is problem-specific and
  will later be folded into an enriched `physics` layer. The module never
  computes an energy. (It may become non-separated later, hence kept out.)
- **No generic training helper** inside the module. The greedy training loop
  lives in the driver/test.
- No relevement (lifting) mode for non-homogeneous BCs in this first version —
  BCs are carried by each axis's `Constraint` and are homogeneous in the beam
  case. Lifting is a later extension.

## Architecture

New package `src/neurom/decompositions/`:

- `__init__.py` — exports `CPPGD`, `Axis`.
- `pgd.py` — the classes below.

### `Axis` (per-factor descriptor)

A lightweight descriptor grouping everything needed to build and interpolate one
factor (one coordinate direction) of the decomposition:

- `name: str` — axis name (e.g. `"space"`, `"E"`), used as key in the separated
  output.
- `topology: Topology`
- `sf: ShapeFunction`
- `mapping` — e.g. `IsoparametricMapping1D(sf)`
- `quad: QuadratureRule`
- `constraint: Constraint` — carries the BCs of this axis (homogeneous here).
- `init_values: torch.Tensor` — initial nodal values used for each new monom on
  this axis.
- `nodes_positions: Field` — the axis mesh node positions (needed to build the
  `Mesh`/`QuadratureContext`).

Rationale: avoids passing many parallel lists to `CPPGD` and extends cleanly to
`l` axes. (Preferred over the raw-list style of `NeuROM`.)

### `CPPGD(nn.Module)`

Constructor: `CPPGD(axes: list[Axis], n_modes_max, n_modes_ini=1)`.

State:

- `self.axes` — the ordered list of `Axis`.
- `self.monoms` — `nn.ModuleList` over modes, each entry an `nn.ModuleList` over
  axes of `TrainableField`. `self.monoms[m][k]` is the monom `w_m^k`, a
  `TrainableField` on `axes[k].topology` with `axes[k].constraint` and
  `axes[k].init_values`.
- `self._contexts` — one `QuadratureContext` per axis (built from the axis
  `Mesh`, `quad`, `mapping`), reused by all modes.
- `self._meshes` — one `Mesh` per axis (for `PointWiseInterpolator` in
  `assemble`).
- `n_modes_max`, `n_modes_truncated` (currently active modes; starts at
  `n_modes_ini`).

Mode management (ported from `NeuROM`):

- `add_mode()` — increment `n_modes_truncated`; zero-out and unfreeze the new
  mode's monoms; freeze the previously active modes' monoms.
- `freeze_mode(m)` / `unfreeze_mode(m)` and per-axis freeze helpers as needed.
- `add_mode_to_optimizer(optim)` — add the newly freed parameters to an existing
  optimizer (`optim.add_param_group`).
- `parameters()` works out of the box (nn.Module) so `.to(device/dtype)` and
  optimizer construction behave normally.

Evaluation:

- `assemble(coords: list[torch.Tensor]) -> torch.Tensor` — full tensor of shape
  `(N_1, ..., N_l)`. For each axis `k`, interpolate every active monom at
  `coords[k]` with `PointWiseInterpolator`; combine over modes with an einsum
  that shares the mode index and produces one output index per axis:
  `u[i_1,...,i_l] = sum_m prod_k W_{m,k}(coords[k])[i_k]`.
  For post-processing, visualization, tests.

- `interpolate_separated() -> dict[str, list[QuadratureAssemblyResult]]` — for
  each axis, a **list indexed by mode**; entry `m` is the interpolation of the
  single monom `w_m^k` at that axis's quadrature points, a
  `QuadratureAssemblyResult` with `u` of shape `(N_e, N_q, u_dim)`, plus the
  (shared) axis geometry `x` and `measure`. Direct per-monom access:
  `sep["space"][m]` is the mode-`m` monom on the space axis. Built by running
  one `QuadratureAssembly` per (mode, axis).

  Rationale for exposing **each monom individually** (rather than a stacked
  `(N_e, N_q, n_modes)` tensor): when the energy is separable the user writes it
  directly as a function of the individual monoms `w_m^k`, so per-monom access
  is the natural interface. Each monom keeps its own autograd link, so
  `jacobian_field` applies per monom (e.g. `d_x w_m^space`).

  Shape convention: the library keeps quadrature points grouped **per element**
  (`N_e` elements x `N_q` points/element) because `measure = w * |det J|` is
  per-element and integration sums over both `N_e` and `N_q`. A monom is scalar
  here (`u_dim = 1`); vector-valued monoms are supported by the same
  `(N_e, N_q, u_dim)` layout.

## Data flow for the external energy (beam example, lives in the test)

Objective:
`J = 1/2 * int_E int_x  E (d_x u)^2 dx dE  -  int_E int_x  f u dx dE`
with `u(x,E) = sum_m S_m(x) g_m(E)`.

1. `sep = model.interpolate_separated()`.
2. Space axis `sep["space"]` is a list over modes; entry `m` gives `S_m`:
   `x_s (N_es,N_qs,1)`, `u_s (N_es,N_qs,1)`, `measure_s`. Derivative
   `d_x S_m = jacobian_field(x_s, u_s)`.
3. `E` axis `sep["E"]` is a list over modes; entry `m` gives `g_m`:
   `x_E` (the E values), `u_E (N_eE,N_qE,1)`, `measure_E`.
4. Separated assembly of the quadratic elastic term by expanding the square,
   written directly in terms of the individual monoms:
   `sum_{m,n} [ int d_x S_m d_x S_n dx ] [ int E g_m g_n dE ]`, plus the load
   term. Cost scales with the sum of axis sizes, not the product.

The test also owns the greedy loop: minimize (LBFGS) at fixed mode count, then
`model.add_mode()` + `model.add_mode_to_optimizer(optim)` between stages.

## Testing

- `assemble` vs `interpolate_separated` consistency on a known separated field.
- Single-mode CP-PGD reduces to the plain FEM beam solution at fixed `E`.
- Reference beam PGD test written by us in
  `tests/integration/test_1d_beam_deflection_PGD_test.py` (the file
  `test_1d_beam_deflection_PGD.py` is left for the user's own implementation as
  an exercise): the assembled `u(x,E)` matches the analytical beam deflection
  `0.5 f (x - x_min)(x - x_max) / E` across the `E` range within the existing
  relative tolerance. This test also demonstrates the external parametric energy
  and the greedy enrichment loop driving `CPPGD`.

## Open questions / future work

- Relevement mode for non-homogeneous BCs.
- Folding the parametric energy into an enriched `physics`/`Term` layer once the
  energy forms stabilize (possibly non-separated).
- r-adaptivity of axis meshes (trainable node positions), as in `NeuROM`.
