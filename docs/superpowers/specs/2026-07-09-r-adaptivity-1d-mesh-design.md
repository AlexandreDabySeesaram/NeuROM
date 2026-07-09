# r-adaptivity for 1D meshes — trainable node positions (design)

Date: 2026-07-09
Issue: [AlexandreDabySeesaram/NeuROM#10 — Introduce r-adaptivity](https://github.com/AlexandreDabySeesaram/NeuROM/issues/10)
Status: approved design, pending implementation plan

## Goal

Let the **positions** of mesh nodes be optimized during training (r-adaptivity),
so interior nodes migrate toward regions of high solution curvature while the
domain endpoints stay fixed. Reproduce the r-adaptivity behaviour of Škardová et
al. (2026), validated on a 1D Poisson problem.

## Background: what already exists

The refactor the issue literally asks for — splitting `Mesh` into a graph
`Topology` (indices) and separate vertex positions — **is already done** in the
current codebase:

- `Topology` (`src/neurom/meshes/topology.py`) holds only `nodes` and
  `connectivity` (indices, registered buffers).
- `Mesh` (`src/neurom/meshes/mesh.py`) holds `topology` + a `nodes_positions`
  provider.
- `nodes_positions` is any `FieldBase` (`full_values()`, `at_elements()`,
  `topology`, `dim`); the isoparametric mapping
  (`src/neurom/geometry/iso_parametric_mapping_1d.py`) is already fully
  differentiable w.r.t. nodal coordinates.
- `IntegrationDomain.update_contexts()` and `QuadratureContext.update()` already
  recompute geometry — but nothing ever calls them.

So the remaining work is **not** the split. It is: (a) a positions provider
whose parameters can be trained *while keeping nodes ordered and endpoints
fixed*, and (b) wiring the per-forward geometry refresh so gradients reach those
parameters.

## Core idea: 1D reparametrization (softplus → cumsum → normalize)

Directly training raw node coordinates (e.g. a `TrainableField` with a
`Dirichlet` constraint pinning the endpoints) is **rejected**: a Dirichlet
constraint fixes the endpoints but does nothing to stop interior nodes from
crossing (element interpenetration, `det J → 0` or sign flip). Instead we
reparametrize so that ordering and endpoints are guaranteed **by construction**.

Let `c ∈ ℝ^{n-1}` be free real parameters (one per interval of an `n`-node 1D
mesh). Physical coordinates are:

```
δ  = softplus(c)                 # (n-1,)  strictly positive increments
x̃  = cat([0], cumsum(δ))         # (n,)    x̃_k = Σ_{i<k} δ_i , strictly increasing
x   = a + (b - a) · x̃ / x̃[-1]    # (n,)    x[0]=a, x[-1]=b exactly
```

- `softplus` keeps every increment positive → `x` strictly increasing → no
  tangling, ever.
- The final affine normalization pins `x[0]=a` and `x[-1]=b` for **any** `a,b`.

This is a **1D-only** technique (a total order on a line). It does not generalize
to 2D/3D, where r-adaptivity needs the general free-position + connectivity
approach (and possibly connectivity recomputation / retriangulation). The
general path in NeuROM (`Topology` + a free-value positions provider) stays
untouched and available; the reparametrization is an *additional*,
dimension-specific provider — never a replacement.

### Note on the notebook source

The logic comes from a prior monolithic notebook class `interpolation1D`, which
fused mesh coordinates, the solution field `u` and its BCs, shape functions, and
the forward interpolation into one object. NeuROM already separates all of these
(`Mesh.nodes_positions`, `TrainableField`+`Dirichlet`, `ShapeFunction` /
`QuadratureContext` / `QuadratureAssembly`), so only the coordinate
reparametrization is new.

One **deliberate correction** to the notebook: prepend `0` (not `a`) before the
cumsum. This matches the stated math (`x̃_k = Σ_{i<k} δ_i`, so `x̃_1 = 0`) and
makes endpoints land exactly on `[a,b]` for any `a`, removing the notebook's
`assert a == 0`.

## Design

### 1. `TrainablePositions1D` — new `FieldBase` subclass

- Location: `src/neurom/fields/trainable_positions_1d.py`, exported from
  `neurom.fields`. Sits beside `Field` / `TrainableField`; the `1D` suffix makes
  the dimensional restriction explicit at every call site. Docstring states
  loudly that this is a 1D-only reparametrization strategy.
- Constructor: `TrainablePositions1D(name, topology, initial_positions)` where
  `initial_positions` is `(n, 1)`, strictly increasing. `a`, `b` are taken from
  the first/last entry.
- Init: recover the raw params from the (possibly non-uniform) input mesh via
  the inverse map — normalize the input increments by `(b-a)` and apply
  `inv_softplus`, so `full_values()` reproduces `initial_positions`.
  Use the numerically-stable inverse `inv_softplus(y) = log(-expm1(-y)) + y`
  (equal to `log(expm1(y))`).
- Interface:
  - `full_values()` → the reparametrized coordinates `(n, 1)` (the forward map
    above). Differentiable in `self.coordinates`.
  - `at_elements()` → `full_values()[topology.connectivity]` (same as other
    fields).
  - `dim` → `1`.
- No runtime round-trip assertion in `__init__` (kept lean); correctness of the
  `inv_softplus`/`softplus` round-trip is covered by a dedicated unit test
  (below).
- `self.coordinates` is an `nn.Parameter` of shape `(n-1,)`. Normalization
  removes one scale DOF, leaving `n-2` effective interior DOFs — matching "2
  endpoints fixed, interior free".

### 2. `Mesh` — first-class owner of trainable nodes

- `Mesh.with_trainable_positions_1d(topology, initial_positions, name="positions")`
  classmethod: builds a `TrainablePositions1D` provider and returns a `Mesh`.
  The plain `Mesh(topology, nodes_positions)` constructor is unchanged for fixed
  meshes.
- `mesh.has_trainable_positions` property → `any(p.requires_grad for p in
  self.nodes_positions.parameters())`. Provider-agnostic (works for any future
  trainable positions provider).

### 3. `FEMModel.forward()` — geometry refresh wiring

At the top of `forward()`, when `self.mesh.has_trainable_positions`, call
`self.integration_domain.update_contexts()` before assembly. This recomputes
`measure` and `quad_pos` from the **current** `full_values()` every forward, so
autograd flows back through normalize → cumsum → softplus to `coordinates`, and
the cached geometry never goes stale. The fixed-mesh path is gated out and
therefore unchanged in behaviour and cost (existing tests unaffected).

## Testing

### Unit tests

`tests/unit/fields/test_trainable_positions_1d.py`:

1. **Round-trip / init fidelity** — for a **uniform** and a **non-uniform**
   strictly-increasing input mesh, `full_values()` reproduces the input to tight
   tolerance (`atol≈1e-5`). This replaces the notebook's in-`__init__`
   assert+print.
2. **Structural invariants** — for arbitrary `coordinates` values (including
   randomly perturbed), `full_values()` is strictly increasing and has
   `x[0]==a`, `x[-1]==b`.
3. **Differentiability** — gradients flow: a scalar of `full_values()` has a
   non-`None`, finite grad w.r.t. `coordinates`.
4. **Interface** — `at_elements()` shape/values match
   `full_values()[connectivity]`; `dim == 1`.

### Integration test: 1D Poisson r-adaptivity

`tests/integration/test_1d_poisson_r_adaptivity.py`.

- **Physics**: 1D Poisson on `[a,b]=[0,1]`, `u(0)=u(1)=0`, energy
  `∫ ½(u')² − f·u` (`ElasticEnergy - LoadPotential`, the existing Poisson weak
  form).
- **Manufactured sharp feature**: `u*(x) = exp(-((x-0.5)/w)²)` (narrow Gaussian
  bump, ~0 at both ends), source `f = -u*''`, `w ≈ 0.05` so the feature is
  under-resolved on a uniform coarse mesh → strong incentive to cluster nodes at
  the center.
- **Physical load under moving nodes**: because nodes move, the load field `f`
  is re-sampled at the *current* node positions each forward (test-harness code),
  so the discrete problem keeps approximating the fixed `u*` rather than drifting
  with node indices.
- **Two runs, same setup**:
  - A (fixed): `nodes_positions = Field`, train displacement only.
  - B (adaptive): `Mesh.with_trainable_positions_1d(...)`, train displacement +
    node params jointly with a single LBFGS over all `model.parameters()`.
- **Assertions**:
  1. Accuracy improves — `‖u_h − u*‖` on a dense sample grid is strictly smaller
     for B than A (discretization-independent metric; primary check).
  2. Nodes adapt — mean `|x_i − 0.5|` over interior nodes decreases for B vs. the
     uniform init; the smallest final element straddles the feature.
  3. Structural sanity — B's final `x[0]==0`, `x[-1]==1`, `diff(x) > 0`
     (guaranteed by construction; asserted as a guard).

Rationale for L2 error (not energy) as the primary metric: the two runs use
slightly different `f`-discretizations, so their potential energies are not
perfectly comparable; `‖u_h − u*‖` against the analytic solution is clean.

## Decisions made (not asked)

- **Joint LBFGS** over {displacement free values, node params} rather than
  alternating minimization — simpler and adequate at this problem size.
- Primary integration assertion is **L2 error**, not energy (see above).

## Known limitations / future work

- Reparametrization is **1D only**; 2D/3D r-adaptivity needs the general
  free-position + connectivity(-recompute) approach, deliberately left in place.
- **CP-PGD interaction (flagged, out of scope here)**: if `TrainablePositions1D`
  is later used *per mode*, different modes live on different meshes → different
  Gauss points. Cross-mode integrals must then be evaluated on a common/union
  set of points (not a naive per-mode Gauss sum). The present test is
  single-field Poisson and does not hit this.
- No explicit minimum-element-size floor beyond `softplus > 0`; extreme
  clustering could still make elements very small. A log-barrier / minimum-size
  term is possible future work.

## Touched files (anticipated)

- new: `src/neurom/fields/trainable_positions_1d.py`
- edit: `src/neurom/fields/__init__.py` (export)
- edit: `src/neurom/meshes/mesh.py` (`with_trainable_positions_1d`,
  `has_trainable_positions`)
- edit: `src/neurom/fem_model.py` (refresh wiring in `forward`)
- new: `tests/unit/fields/test_trainable_positions_1d.py`
- new: `tests/integration/test_1d_poisson_r_adaptivity.py`
- edit: `CHANGELOG.md`
