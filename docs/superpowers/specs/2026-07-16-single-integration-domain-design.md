# One IntegrationDomain for the whole problem — design spec

**Date:** 2026-07-16
**Branch:** `pgd_addition_solal`
**Amends:** `docs/superpowers/specs/2026-07-13-neurom-model-separated-domain-design.md`
(removes the `SeparatedDomain` it introduced; keeps `NeuROMModel`, `directory()`,
`evaluate`/`assemble`).

## Context

The 2026-07-13 work gave `CPPGD` its own `SeparatedDomain` and a `fill()` method
on the `TensorDecomposition` contract. The decomposition therefore owns a private
interpolation pipeline: it builds its own `Mesh` and `QuadratureContext`
(`pgd.py`), wraps them in its own domain, and `NeuROMModel.forward` calls
`decomposition.fill(layout)` instead of `domain.interpolate_all(layout)`.

**Anything that is not a monom is excluded from that pipeline by construction.**
The beam test registers a constant load in the layout but nothing ever
interpolates it, and reading it back fails:

```
RuntimeError: Field 'load' registered but not yet interpolated.
```

(`FieldLayout.add` fills `_fields`; `__getitem__` reads `_interp`, which only
`update()` — i.e. an interpolation — fills.) A user cannot fix this from outside:
building the load's assembly requires the *same* `QuadratureContext` object as
the space monoms (so `x` and `measure` align and the context is deduped), and
that object is `pgd._contexts[0]` — private.

This is a deviation from the library's own idiom. `scripts/design_poc/main.py`
already solves exactly this problem, in the shape we want:

```python
ctx = QuadratureContext(mesh, quad, mapping)
assembly_u = QuadratureAssembly(ctx, sf, u)
assembly_f = QuadratureAssembly(ctx, sf, f)      # the load, sharing u's ctx
domain = IntegrationDomain([assembly_u, assembly_f])
model = FEMModel(mesh, field_layout, integration_domain=domain, loss=physics_loss)
```

One assembly per field, one domain for the problem, built explicitly outside.
This spec returns `CPPGD` to that idiom.

## What `SeparatedDomain` actually buys

Its only difference from `IntegrationDomain` is skipping inactive mode-blocks.
That filtering is **not** required for correctness: the energy reads modes
through `directory()`, which already truncates to active modes, so interpolating
an inactive monom would only write a layout entry nobody reads. The filtering is
a **compute saving** — but a real one, and one worth keeping: `QuadratureContext`
sets `_xi_ref.requires_grad_(True)`, so every interpolation builds autograd graph
(that is what lets `jacobian_field` compute `du/dx`). Interpolating inactive
modes would retain that graph under `retain_graph=True`, costing memory as well
as flops. The saving must survive; the class need not.

## Goals

- One `IntegrationDomain` per problem, holding the decomposition's assemblies
  **and** every other field the energy reads (loads, sources, coefficients).
- Keep zero wasted interpolation of inactive PGD modes.
- `NeuROMModel` and `FEMModel` converge on the same pipeline:
  `integration_domain.interpolate_all(layout)` then energy/loss.
- A user builds `Axis` objects and hands them to `CPPGD`; no manual `Mesh` /
  `QuadratureContext` construction.

## Non-goals

- No 2-D axes, no sub/super-parametric elements, no reusable energy `Term`, no
  Tucker/TT. This spec only moves responsibilities; see *Forward compatibility*
  for why it does not block them.
- No change to the separable energy's math.

## Design decisions (resolved forks)

| Fork | Decision |
|---|---|
| Scope | **Decomposition stops interpolating.** `fill()` leaves the `TensorDecomposition` contract; `NeuROMModel` takes an `integration_domain` like `FEMModel`. |
| Truncation mechanism | **`active` bool buffer on `QuadratureAssembly`**; `IntegrationDomain.interpolate_all` skips inactive. Rejected: a dynamic `domain.add(assembly)` (couples `CPPGD` to the domain, `state_dict` keys appear mid-training); lazy pull-based interpolation in `FieldLayout` (inverts the library's push model, too large a blast radius). |
| `n_modes_truncated` | **Derived** by counting leading active mode-blocks. No integer buffer — the flags are the single source of truth. |
| Context construction | **`Axis.__post_init__` builds and exposes `.mesh` and `.context`.** The `Axis` public signature is unchanged. |
| `grow()` | Moves to `CPPGD.add_mode()`, which flips the new block's flags. |

## `active` vs `requires_grad` vs optimizer membership

Three distinct flags, one per phase of an iteration. They are **nested**
(`optimized ⊆ requires_grad ⊆ active`), not equal, and the nesting is forced by
autograd: a tensor absent from the forward is disconnected (`.grad` stays
`None`), and Adam skips a `None` grad.

| Flag | Held by | Phase | Question |
|---|---|---|---|
| `active` | the assembly | forward | is this field evaluated at quadrature points? |
| `requires_grad` | the tensor | backward | do we want `d(energy)/d(this tensor)`? |
| `param_group` membership | the optimizer | step | does the optimizer update this tensor? |

Each ring has a real inhabitant in this library:

- **`active` only** — the load. `Field` stores values via `register_buffer`, so
  `requires_grad=False` permanently, yet it must be interpolated.
- **`active` + `requires_grad`, not optimized** — `_xi_ref`
  (`quadrature_context.py:37`): a bare tensor, in no optimizer, requiring grad
  solely so `jacobian_field` can differentiate w.r.t. `x`.
- **all three** — the current mode's monoms.

**Therefore `active` cannot be derived from `requires_grad`.** Two populations
share `requires_grad=False` while differing on `active`: an unborn mode (skip it)
and a *frozen converged mode* (interpolate it — `directory()` still lists it, and
the energy's cross terms `Σ_{m,n}` need `grad(u_m)` for frozen `m`). The load
sits in the outer ring regardless. No function of `requires_grad` separates
"frozen but read" from "not yet born".

**Mode lifecycle** (monotone in `active`, oscillating in `requires_grad`):

```
born         active=False, requires_grad=False   # does not exist for the problem
add_mode(m)  active=True,  requires_grad=True    # current unknown
freeze_mode  active=True,  requires_grad=False   # contributes as frozen data
```

**Illegal state:** `active=False` with `requires_grad=True`. It does not raise —
it silently does nothing (`.grad is None`, Adam skips), i.e. a mode you believe
you are training that never moves. `add_mode` activates **before** unfreezing so
the transition never passes through it; a unit test asserts the invariant.

Frozen modes keep `requires_grad=False`, not because "they are not optimized"
(`_xi_ref` disproves that framing) but because we genuinely do not want
`d(energy)/d(w_m)`: autograd then prunes those branches instead of walking them
and allocating `.grad`. Note the library's convention — for `Parameter`s,
`requires_grad ⟺ optimized` — which the beam test relies on
(`[p for p in model.parameters() if p.requires_grad]`). It is a convention, not a
PyTorch semantic. Nouy's *update* step (re-optimizing all modes together) would
flip frozen modes back to `requires_grad=True`, and the middle ring becomes
useful.

## Components

### 1. `QuadratureAssembly` — `src/neurom/interpolation/quadrature_assembly.py`

```python
def __init__(self, context, sf, field, active: bool = True):
    ...
    self.register_buffer("active", torch.tensor(bool(active)))

def activate(self):
    """Mark this assembly for interpolation (in-place, keeps the buffer identity)."""
    self.active.fill_(True)
```

`active=True` by default: every existing call site (POC, FEM tests) keeps its
behaviour. Buffer, not plain attribute → round-trips through `state_dict`.
No `deactivate()`: the PGD lifecycle is monotone, and non-PGD fields pass
`active=False` at construction if they ever need it (YAGNI).

### 2. `IntegrationDomain` — `src/neurom/interpolation/integration_domain.py`

```python
def interpolate_all(self, field_layout):
    for assembly in self.assemblies:
        if not bool(assembly.active):
            continue
        field_layout.update(assembly.field, assembly.interpolate())
```

Nothing else changes: context dedup and `update_contexts()` are untouched, and
now cover the whole problem (relevant once axis nodes become trainable —
r-adaptivity). Drop the dead `from neurom.field_layout import FieldLayout` import
inside the method.

The flag is **per assembly** — the finest granularity. `SeparatedDomain` carved
"one block = one mode" into the interpolation layer; the flag does not, so a
future format whose factors group differently (Tucker core, TT rank, a factor
spanning two axes) needs no change here. Mode-block structure stays private to
`CPPGD`.

### 3. `SeparatedDomain` — **deleted**

Delete `src/neurom/interpolation/separated_domain.py` and its export from
`src/neurom/interpolation/__init__.py`.

### 4. `Axis` — `src/neurom/decompositions/pgd.py`

`nodes_positions + mapping + quad` are exactly a `QuadratureContext`'s
ingredients, and the `Axis` already holds all three. It builds the context
instead of handing the parts to `CPPGD` to rebuild privately.

```python
def __post_init__(self):
    self.mesh = Mesh(self.topology, self.nodes_positions)
    self.context = QuadratureContext(self.mesh, self.quad, self.mapping)
```

The public signature is **unchanged**; `.mesh` and `.context` are derived
attributes. Deriving `self.topology` from `nodes_positions` (as today) keeps the
`Topology`-identity check inside `Mesh` satisfied by construction.

**Invariant — the `Axis` builds the context, never the mapping.** The mapping
stays an injected argument. Building it (`mapping = IsoparametricMapping(self.sf)`)
would weld the geometry's shape function to the field's and close the door on
sub/super-parametric elements. See *Forward compatibility*.

**Known trade-off:** the `Axis` stops being an inert descriptor and becomes the
owner of live `nn.Module`s with computed geometry. Constructing one is no longer
free (it maps, inverse-maps and computes measures) — the same work as today, just
moved out of `CPPGD.__init__`. And mutating a field afterwards
(`axis.quad = ...`) will not rebuild the context. Documented as an invariant
rather than enforced: freezing the dataclass would force `object.__setattr__`
calls in `__post_init__`, costing more readability than the safety is worth.

### 5. `CPPGD` — `src/neurom/decompositions/pgd.py`

**Reads contexts, builds none.** `self._meshes` disappears; `self._contexts`
becomes a reference list kept solely for `nn.Module` registration
(`.to(device)`, `state_dict`), so a unit-tested `CPPGD` carries its axes'
geometry even without a domain:

```python
self._contexts = nn.ModuleList([a.context for a in self.axes])   # registration only
```

**Owns its assemblies, no domain.**

```python
n_ini = min(n_modes_ini, n_modes_max)
self._assemblies = nn.ModuleList([
    nn.ModuleList([
        QuadratureAssembly(a.context, a.sf, self.monoms[m][k], active=(m < n_ini))
        for k, a in enumerate(self.axes)
    ])
    for m in range(self.n_modes_max)
])
```

Construction order: contexts → monoms → assemblies (the flags carry `n_ini`) →
`freeze_all()` → `unfreeze_mode(m)` for `m < n_modes_truncated` (which reads the
flags). `_assemblies` is a `ModuleList` of `ModuleList` so the `active` buffers
round-trip under `CPPGD` too; the assemblies are the same objects the domain
holds, and `nn.Module` dedups parameters by identity (`state_dict` emits both key
paths, as it already does for the monoms — harmless).

**New public accessor** — the seam the domain is built from:

```python
def assemblies(self):
    """Flat list of this decomposition's QuadratureAssembly, one per monom."""
    return [a for block in self._assemblies for a in block]
```

**`n_modes_truncated` derived from the flags** (single source of truth; the
2026-07-13 buffer had its home on `SeparatedDomain` and loses it here):

```python
@property
def n_modes_truncated(self) -> int:
    """Number of active modes: the leading run of active mode-blocks.

    Active blocks are contiguous from index 0 by the greedy lifecycle — a mode,
    once activated, is never deactivated.
    """
    n = 0
    for block in self._assemblies:
        if not bool(block[0].active):
            break
        n += 1
    return n
```

**`add_mode`** — activate, then unfreeze (never transit the illegal state):

```python
def add_mode(self):
    m = self.n_modes_truncated
    if m >= self.n_modes_max:
        raise RuntimeError("Cannot add a mode: all modes are already active.")
    for assembly in self._assemblies[m]:
        assembly.activate()
    self.unfreeze_mode(m)
    return m
```

The existing docstring rationale stands unchanged: the new mode keeps its
`Axis.init_values` seed rather than being zeroed (an all-zero mode is a
stationary point).

**Remove `fill()`** and `self.domain`. **`evaluate` / `assemble`**: replace
`self._meshes[k]` with `axis.mesh` — same objects, public path. `register_into`,
`directory`, `freeze_*` / `unfreeze_*`, `add_mode_to_optimizer` unchanged.

### 6. `NeuROMModel` — `src/neurom/neurom_model.py`

```python
def __init__(self, field_layout, decomposition, integration_domain, energy):
    super().__init__()
    self.field_layout = field_layout
    self.decomposition = decomposition
    self.integration_domain = integration_domain
    self.energy = energy
    decomposition.register_into(field_layout)

def forward(self, coords=None):
    if self.training:
        self.integration_domain.interpolate_all(self.field_layout)
        return self.field_layout
    if coords is None:
        raise ValueError(
            "eval forward requires coords: a (P, n_axes) tensor, one point per row."
        )
    return self.decomposition.evaluate(coords)
```

Argument order mirrors `FEMModel(mesh, field_layout, integration_domain, loss)`.
No `mesh` argument: a decomposition has one per axis, and the domain already
reaches them through its contexts. `assemble` still delegates to the
decomposition.

### 7. `TensorDecomposition` ABC — `src/neurom/decompositions/base.py`

Drop `fill`. The contract becomes `register_into` (setup) + `evaluate` /
`assemble` (inference). Interpolation is the domain's job, for every field alike.

## Target user code

```python
axis_space = Axis(name="space", nodes_positions=nodes_positions_space, sf=sf,
                  mapping=mapping, quad=quad,
                  constraint=Dirichlet(nodes=[0, N - 1], values_imposed=torch.zeros(2, 1)),
                  init_values=u_init)
axis_E     = Axis(name="E", nodes_positions=nodes_positions_E, sf=sf, mapping=mapping,
                  quad=quad, constraint=NoConstraint(), init_values=E_init)

pgd = CPPGD(axes=[axis_space, axis_E], n_modes_max=3, n_modes_ini=1, name="pgd")

load = field_layout.add(Field(name="load", topology=topology_space,
                              values=load_value * torch.ones(N_space, 1)))
assembly_load = QuadratureAssembly(axis_space.context, sf, load)   # space monoms' ctx

domain = IntegrationDomain([*pgd.assemblies(), assembly_load])
model  = NeuROMModel(field_layout, pgd, domain, energy=lambda out: energy(out, pgd, "load"))
```

Line for line the POC's shape. The energy's math is untouched: `load_field.u` now
holds the load sampled at the space quadrature points, which is what
`inner(load_f, u[m]) * J_u[m]` already assumes.

## Forward compatibility

Checked against the planned extensions (2-D axes, vector monoms, a monom whose
dimension differs from its input variable's):

- **The `mapping` is already purely geometric.** `IsoparametricMapping1D.map /
  inverse_map / det_jacobian` only ever see `x_nodes =
  mesh.nodes_positions.at_elements()`. The monom is interpolated by `sf.N(xi)`
  and the einsum `"en...,eqn...->eq..."` whose `...` *is* the field dimension.
  So "the mapping used by the `Axis`" vs "by the monom" needs no separation: the
  monom has no mapping, it has an `sf`.
- **`u(x, F) = μ(x) f(F)` already works.** A 2-D `F` mesh with a scalar `f` is
  `init_values` of shape `(N_F, 1)`; a 2-D `x` mesh with a vector `μ` is
  `(N_x, 2)`. Same `sf`, same context, same assembly. The constructor guard
  `sum(a.init_values.shape[1] > 1) > 1` already encodes "at most one vector
  factor per mode" — literally this case.
- **The refactor makes the split explicit.** After `__post_init__` the `Axis`'s
  fields fall into two disjoint groups: what enters the context is geometry
  (`nodes_positions`, `mapping`, `quad`); everything else describes the monom
  (`sf`, `init_values`, `constraint`). Today it is a flat bag of seven where
  nothing says which serves what.
- **2-D axes are orthogonal.** Neither the flag, nor the single domain, nor the
  removal of `fill()` mentions dimension. The 2-D work (2-D quadrature rules,
  `sf`, mapping, `reference_coordinates`) is independent.
- **Remaining coupling, unchanged by this spec:** sub/super-parametric elements
  need the *field's* topology to differ from the *geometry's* (a Q2 field has
  more nodes than a Q1 geometry), while `Axis.topology` derives from
  `nodes_positions` and serves both. This spec neither creates nor removes that.
  `μ(x) f(F)` does not hit it.

## Testing strategy

**`tests/unit/interpolation/test_separated_domain.py` → `test_integration_domain.py`**
- Inactive assembly is not interpolated: reading its field from the layout raises
  `RuntimeError`; the active one reads back fine.
- `active` defaults to `True` (existing call sites unaffected).
- `activate()` flips it and the next `interpolate_all` includes the assembly.
- Contexts stay deduped; `update_contexts()` recomputes each unique context once.
- Mixed domain: PGD assemblies + a plain data-field assembly sharing one context.

**`tests/unit/decompositions/test_pgd.py`**
- Replace `test_cppgd_owns_separated_domain_synced_with_truncation` with:
  `n_modes_truncated` == number of leading active blocks, and it grows after
  `add_mode()`.
- `add_mode` at capacity raises `RuntimeError`.
- **Invariant test:** no parameter with `requires_grad=True` belongs to an
  inactive assembly — over the whole greedy sequence (initial state,
  `freeze_mode(0)` + `add_mode()`, capacity).
- **Frozen-but-active:** after `freeze_mode(0); add_mode()`, mode 0's assemblies
  are still `active` and its monoms still land in the layout.
- `Axis.__post_init__`: `.mesh` / `.context` exist; `axis.mesh.topology is
  axis.topology`; two `Axis` objects yield distinct contexts.
- `CPPGD.assemblies()`: flat, `n_modes_max * n_axes` long, and each element's
  `context is axes[k].context` (the sharing the load depends on).
- Drop the `fill()` tests; the fake `TensorDecomposition` loses its `fill`.
- `NeuROMModel`: train `forward()` runs `interpolate_all` on the injected domain
  and returns the layout; eval `forward(coords)` unchanged.

**`tests/integration/test_1d_beam_deflection_PGD_human.py`**
- Add `assembly_load` + the explicit `IntegrationDomain`; the test must now run
  to completion (it currently raises on `field_layout["load"]`) and the rank-2
  fit must still match the analytical target.

## Edge cases

- `add_mode()` at capacity → `RuntimeError` from `CPPGD` (no domain to raise it).
- A domain built without a decomposition's assemblies → its monoms are never
  interpolated → `RuntimeError: registered but not yet interpolated` at energy
  time. Same failure mode as today's load: loud, not silent.
- An assembly listed twice in one domain → interpolated twice, last write wins;
  wasteful but correct. Not guarded (YAGNI).
- `NeuROMModel` given a non-fresh `field_layout` → `FieldLayout.add` raises
  `ValueError` on the duplicate monom name, as today.

## Out of scope / deferred

- Reusable `SeparablePGDEnergy` `Term` (the energy stays test-local).
- 2-D axes; sub/super-parametric elements; Tucker/TT.
- Nouy's *update* step (re-optimizing all modes together).

## Changelog

Qualifies for a `CHANGELOG.md` entry (API + structural change): `SeparatedDomain`
removed, `fill()` off the `TensorDecomposition` contract, `NeuROMModel` takes an
`integration_domain`, `Axis` owns its `QuadratureContext`. Link back to this
spec.
