# Single IntegrationDomain per problem — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one `IntegrationDomain` interpolate every field of a problem — the PGD monoms *and* other fields like loads — by removing `SeparatedDomain`, moving mode truncation to an `active` flag on `QuadratureAssembly`, taking `fill()` off the decomposition contract, and giving `NeuROMModel` an injected `integration_domain` like `FEMModel`.

**Architecture:** Truncation becomes a per-assembly `active` buffer that `IntegrationDomain.interpolate_all` honours (skip inactive). `Axis` builds and exposes its own `QuadratureContext`/`Mesh`, so a load can share the monoms' context. `CPPGD` stops owning a domain and instead exposes `assemblies()`; the caller builds one `IntegrationDomain([*pgd.assemblies(), load_assembly])`. `NeuROMModel.forward` calls `integration_domain.interpolate_all(layout)`, converging with `FEMModel`.

**Tech Stack:** Python ≥3.12, PyTorch (`nn.Module`, buffers, autograd), pytest. Run tests with `.venv/bin/python -m pytest`.

## Global Constraints

- Python ≥3.12; tests set `torch.set_default_dtype(torch.float32)`.
- Run tests via `.venv/bin/python -m pytest` (pytest is not on the base interpreter). Headless plotting: prefix `MPLBACKEND=Agg`.
- Monom field names are `f"{name}_dim{axis.name}_mode{m}"` (unchanged).
- CP admits **at most one vector-valued factor per mode** — guarded at construction (unchanged).
- Every stateful flag is a **registered buffer** so it round-trips through `state_dict`.
- `QuadratureAssembly(..., active=True)` defaults to active: no existing call site (POC, FEM tests) changes behaviour.
- Mode lifecycle is monotone in `active`: a mode, once activated, is never deactivated. `add_mode` **activates before unfreezing** so the transition never passes through the illegal `active=False, requires_grad=True` state.
- End git commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 1: `active` flag on `QuadratureAssembly` + skip in `IntegrationDomain`

**Files:**
- Modify: `src/neurom/interpolation/quadrature_assembly.py`
- Modify: `src/neurom/interpolation/integration_domain.py`
- Test: `tests/unit/interpolation/test_integration_domain.py` (new)

**Interfaces:**
- Produces:
  - `QuadratureAssembly(context, sf, field, active: bool = True)` with a bool buffer `active` and a method `activate() -> None` (sets it True in place).
  - `IntegrationDomain.interpolate_all(field_layout)` skips assemblies whose `active` is False.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/interpolation/test_integration_domain.py`:

```python
import pytest
import torch

from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.constraints import NoConstraint
from neurom.field_layout import FieldLayout
from neurom.interpolation import QuadratureContext, QuadratureAssembly, IntegrationDomain

torch.set_default_dtype(torch.float32)


def _ctx(n=4):
    coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topo = Topology(nodes, elements)
    positions = Field(name="x", topology=topo, values=coords)
    sf = LinearSegment()
    mesh = Mesh(topology=topo, nodes_positions=positions)
    ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(sf))
    return ctx, topo, sf


def _field(name, topo, n=4):
    return TrainableField(
        name=name, topology=topo, init_values=torch.ones(n, 1), constraint=NoConstraint()
    )


def _layout(fields):
    layout = FieldLayout()
    for f in fields:
        layout.add(f)
    return layout


def test_active_defaults_true():
    ctx, topo, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("w", topo))
    assert bool(a.active) is True


def test_active_is_a_buffer():
    ctx, topo, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("w", topo))
    assert "active" in dict(a.named_buffers())


def test_inactive_assembly_is_not_interpolated():
    ctx, topo, sf = _ctx()
    fa = _field("wa", topo)
    fb = _field("wb", topo)
    a = QuadratureAssembly(ctx, sf, fa, active=True)
    b = QuadratureAssembly(ctx, sf, fb, active=False)
    domain = IntegrationDomain([a, b])
    layout = _layout([fa, fb])
    domain.interpolate_all(layout)
    assert layout[fa.name].u.shape[-1] == 1        # active -> interpolated
    with pytest.raises(RuntimeError):              # inactive -> not interpolated
        _ = layout[fb.name]


def test_activate_makes_next_interpolation_include_it():
    ctx, topo, sf = _ctx()
    fb = _field("wb", topo)
    b = QuadratureAssembly(ctx, sf, fb, active=False)
    domain = IntegrationDomain([b])
    layout = _layout([fb])
    b.activate()
    domain.interpolate_all(layout)
    assert layout[fb.name].u.shape[-1] == 1


def test_contexts_deduplicated_across_assemblies():
    ctx, topo, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("wa", topo))
    b = QuadratureAssembly(ctx, sf, _field("wb", topo))
    domain = IntegrationDomain([a, b])
    assert len(domain._contexts) == 1
    assert domain._contexts[0] is ctx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/interpolation/test_integration_domain.py -v`
Expected: FAIL — `QuadratureAssembly` has no `active` / `activate`; `TypeError` on the `active=` kwarg.

- [ ] **Step 3: Add the `active` buffer and `activate()` to `QuadratureAssembly`**

In `src/neurom/interpolation/quadrature_assembly.py`, change the constructor signature and body. Current:

```python
    def __init__(self, context: QuadratureContext, sf: ShapeFunction, field: FieldBase):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)
```

Replace with (add `import torch` at the top of the file if not present — it is):

```python
    def __init__(
        self,
        context: QuadratureContext,
        sf: ShapeFunction,
        field: FieldBase,
        active: bool = True,
    ):
        super().__init__()
        self.context = context
        self.field = field
        self.sf = sf
        self._field_interpolator = FieldInterpolator(self.sf, self.field)
        # Whether interpolate_all should evaluate this assembly. A registered
        # buffer so it round-trips through state_dict. Monotone for PGD modes
        # (activated, never deactivated); see the single-IntegrationDomain spec.
        self.register_buffer("active", torch.tensor(bool(active)))

    def activate(self) -> None:
        """Mark this assembly for interpolation (in place; keeps buffer identity)."""
        self.active.fill_(True)
```

`torch` is already imported at the top of this file.

- [ ] **Step 4: Skip inactive assemblies in `IntegrationDomain.interpolate_all`**

In `src/neurom/interpolation/integration_domain.py`, replace `interpolate_all`. Current:

```python
    def interpolate_all(self, field_layout: "FieldLayout"):
        from neurom.field_layout import FieldLayout

        # Interpolate all required fields and update() their values in FieldLayout
        for assembly in self.assemblies:
            result = assembly.interpolate()
            field_layout.update(assembly.field, result)
```

Replace with (drops the dead `FieldLayout` import; adds the skip):

```python
    def interpolate_all(self, field_layout):
        # Interpolate every *active* field and update() its values in the layout.
        # Inactive assemblies (e.g. not-yet-enriched PGD modes) are skipped so no
        # autograd graph is built for them.
        for assembly in self.assemblies:
            if not bool(assembly.active):
                continue
            field_layout.update(assembly.field, assembly.interpolate())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/interpolation/test_integration_domain.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Verify the existing suite still passes (active defaults True)**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS — no existing test regresses (`SeparatedDomain` inherits the new `interpolate_all` but its blocks are all-active by construction).

- [ ] **Step 7: Commit**

```bash
git add src/neurom/interpolation/quadrature_assembly.py \
        src/neurom/interpolation/integration_domain.py \
        tests/unit/interpolation/test_integration_domain.py
git commit -m "$(cat <<'EOF'
feat(interpolation): active flag on QuadratureAssembly, skipped by domain

QuadratureAssembly gains an `active` bool buffer (default True) and
activate(); IntegrationDomain.interpolate_all skips inactive assemblies.
Truncation moves out of SeparatedDomain toward a per-assembly flag.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Axis` builds and exposes its own `Mesh` and `QuadratureContext`

**Files:**
- Modify: `src/neurom/decompositions/pgd.py` (the `Axis` dataclass only)
- Test: `tests/unit/decompositions/test_pgd.py` (add tests)

**Interfaces:**
- Consumes: `QuadratureAssembly(..., active=...)` from Task 1 (not directly here, but same module family).
- Produces: after construction, `Axis` exposes `axis.mesh` (a `Mesh`) and `axis.context` (a `QuadratureContext`). Public signature (`name, nodes_positions, sf, mapping, quad, constraint, init_values`) and `axis.topology` are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/decompositions/test_pgd.py` (helpers `make_axis` / `make_two_axes` already exist in that file):

```python
def test_axis_builds_mesh_and_context():
    from neurom.meshes import Mesh
    from neurom.interpolation.quadrature_context import QuadratureContext

    axis = make_axis()
    assert isinstance(axis.mesh, Mesh)
    assert isinstance(axis.context, QuadratureContext)
    # Mesh identity: the context's mesh is the axis mesh, built on the axis topology.
    assert axis.mesh.topology is axis.topology
    assert axis.mesh.nodes_positions is axis.nodes_positions


def test_two_axes_have_distinct_contexts():
    space, para = make_two_axes()
    assert space.context is not para.context
    assert space.mesh is not para.mesh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py::test_axis_builds_mesh_and_context -v`
Expected: FAIL — `Axis` has no attribute `mesh`.

- [ ] **Step 3: Add `__post_init__` to `Axis`**

In `src/neurom/decompositions/pgd.py`, the `Axis` dataclass currently ends with the `topology` property. Add a `__post_init__` right after the fields (before or after the `topology` property is fine; place it after the property). Insert:

```python
    def __post_init__(self):
        # Build the interpolation geometry once, here, so it is a first-class
        # attribute other fields (e.g. a load) can share via `axis.context`.
        # INVARIANT: the Axis builds the *context*, never the *mapping* — the
        # mapping stays injected so a future sub/super-parametric element can use
        # a geometry shape function distinct from the field's `sf`.
        self.mesh = Mesh(self.topology, self.nodes_positions)
        self.context = QuadratureContext(self.mesh, self.quad, self.mapping)
```

`Mesh` and `QuadratureContext` are already imported at the top of `pgd.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py::test_axis_builds_mesh_and_context tests/unit/decompositions/test_pgd.py::test_two_axes_have_distinct_contexts -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify no regression (CPPGD still builds its own duplicates for now)**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py -q`
Expected: PASS — `CPPGD` still builds its own meshes/contexts internally; the new `Axis` attributes are additive and unused yet.

- [ ] **Step 6: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "$(cat <<'EOF'
feat(decompositions): Axis builds and exposes its Mesh and QuadratureContext

Axis.__post_init__ constructs the interpolation geometry once so non-monom
fields (loads) can share axis.context. Public signature unchanged; the
mapping stays injected (invariant for sub/super-parametric elements).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `CPPGD` reads axis contexts, owns flagged assemblies, derives truncation

This task makes `CPPGD` stop using `SeparatedDomain` internally. It keeps a
**temporary** `fill()` (a direct loop over active assemblies) so `NeuROMModel`
— unchanged in this task — keeps working and the suite stays green. Task 4
removes `fill()` and injects the domain into `NeuROMModel`.

**Files:**
- Modify: `src/neurom/decompositions/pgd.py` (`CPPGD` only)
- Test: `tests/unit/decompositions/test_pgd.py` (rewrite affected tests)

**Interfaces:**
- Consumes: `axis.context` / `axis.mesh` (Task 2); `QuadratureAssembly(..., active=...)` and `.activate()` (Task 1).
- Produces:
  - `CPPGD.assemblies() -> list[QuadratureAssembly]` — flat, `n_modes_max * n_axes` long, in mode-major then axis order; each element's `.context is axes[k].context`.
  - `CPPGD.n_modes_truncated` — property; number of leading active mode-blocks.
  - `CPPGD.add_mode() -> int` — activates the next block's assemblies then unfreezes its monoms; raises `RuntimeError` at capacity.
  - `CPPGD.fill(field_layout)` — temporary; interpolates active monoms into the layout.
  - `CPPGD._assemblies` — `nn.ModuleList[nn.ModuleList[QuadratureAssembly]]`, mode-blocked.
  - `CPPGD` no longer has `self.domain`, `self._meshes`; `self._contexts` is a reference `ModuleList` (`self._contexts[k] is axes[k].context`).

- [ ] **Step 1: Rewrite the CPPGD constructor internals**

In `src/neurom/decompositions/pgd.py`, replace the block that builds meshes, contexts, monoms, and the domain. Current (roughly lines 101–145):

```python
        # One Mesh + QuadratureContext per axis, shared across modes.
        self._meshes = nn.ModuleList(
            [Mesh(a.topology, a.nodes_positions) for a in self.axes]
        )
        self._contexts = nn.ModuleList(
            [
                QuadratureContext(mesh, a.quad, a.mapping)
                for mesh, a in zip(self._meshes, self.axes)
            ]
        )

        # Grid of monoms: modes x axes of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{self.name}_dim{a.name}_mode{m}",
                            topology=a.topology,
                            init_values=a.init_values,
                            constraint=a.constraint,
                        )
                        for a in self.axes
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Truncation-aware domain: one assembly per monom, grouped by mode.
        mode_blocks = [
            [
                QuadratureAssembly(self._contexts[k], a.sf, self.monoms[m][k])
                for k, a in enumerate(self.axes)
            ]
            for m in range(self.n_modes_max)
        ]
        self.domain = SeparatedDomain(
            mode_blocks, n_active_modes=min(n_modes_ini, n_modes_max)
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(self.n_modes_truncated):
            self.unfreeze_mode(m)
```

Replace with:

```python
        # Contexts are built and owned by the axes (Axis.__post_init__). Keep a
        # reference ModuleList only so nn.Module registers them (.to(device),
        # state_dict); dedup by identity happens in IntegrationDomain.
        self._contexts = nn.ModuleList([a.context for a in self.axes])

        # Grid of monoms: modes x axes of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{self.name}_dim{a.name}_mode{m}",
                            topology=a.topology,
                            init_values=a.init_values,
                            constraint=a.constraint,
                        )
                        for a in self.axes
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # One QuadratureAssembly per monom, grouped into mode-blocks. The leading
        # `n_ini` blocks start active; the rest inactive. `active` is the single
        # source of truth for truncation (n_modes_truncated counts leading active
        # blocks) — no separate counter to keep in sync.
        n_ini = min(n_modes_ini, n_modes_max)
        self._assemblies = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QuadratureAssembly(
                            a.context, a.sf, self.monoms[m][k], active=(m < n_ini)
                        )
                        for k, a in enumerate(self.axes)
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(self.n_modes_truncated):
            self.unfreeze_mode(m)
```

- [ ] **Step 2: Replace `n_modes_truncated`, add `assemblies()`, rewrite `add_mode`, rewrite `fill`**

Replace the `n_modes_truncated` property. Current:

```python
    @property
    def n_modes_truncated(self) -> int:
        """Number of currently-active modes (single source of truth: the domain)."""
        return int(self.domain.n_active_modes)
```

Replace with:

```python
    @property
    def n_modes_truncated(self) -> int:
        """Number of active modes: the leading run of active mode-blocks.

        Single source of truth is the assemblies' `active` flags. Active blocks
        are contiguous from index 0 because the greedy lifecycle is monotone —
        a mode, once activated, is never deactivated.
        """
        n = 0
        for block in self._assemblies:
            if not bool(block[0].active):
                break
            n += 1
        return n

    def assemblies(self):
        """Flat list of this decomposition's QuadratureAssembly, one per monom.

        Mode-major then axis order. The seam a caller uses to build the shared
        ``IntegrationDomain([*pgd.assemblies(), other_assembly])``.
        """
        return [a for block in self._assemblies for a in block]
```

Replace `add_mode`. Current:

```python
    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Activates the next mode-block in the domain (trainable) without touching
        the freeze state of the currently-active modes. Returns the index of the
        newly-activated mode. Raises RuntimeError at capacity.

        The new mode keeps its ``Axis.init_values`` seed rather than being zeroed:
        an all-zero mode is a stationary point of the energy (every gradient
        component is proportional to the *other* factor, so both stay locked at
        0), which never takes off under a gradient optimizer. A non-zero
        parametric seed lets the linear load term drive the enrichment.
        """
        new = self.domain.grow()
        self.unfreeze_mode(new)
        return new
```

Replace with:

```python
    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Activates the next mode-block's assemblies, then unfreezes its monoms
        (activate-before-unfreeze, so the mode never passes through the illegal
        active=False/requires_grad=True state). Leaves the freeze state of the
        currently-active modes untouched. Returns the new mode index. Raises
        RuntimeError at capacity.

        The new mode keeps its ``Axis.init_values`` seed rather than being zeroed:
        an all-zero mode is a stationary point of the energy (every gradient
        component is proportional to the *other* factor, so both stay locked at
        0), which never takes off under a gradient optimizer. A non-zero
        parametric seed lets the linear load term drive the enrichment.
        """
        m = self.n_modes_truncated
        if m >= self.n_modes_max:
            raise RuntimeError("Cannot add a mode: all modes are already active.")
        for assembly in self._assemblies[m]:
            assembly.activate()
        self.unfreeze_mode(m)
        return m
```

Replace `fill`. Current:

```python
    def fill(self, field_layout):
        """Interpolate every active monom and ``update`` it in the layout.

        Delegates to the truncation-aware :class:`SeparatedDomain`: only active
        modes are interpolated. CP analogue of
        ``IntegrationDomain.interpolate_all``.
        """
        self.domain.interpolate_all(field_layout)
```

Replace with (temporary — removed in Task 4; keeps `NeuROMModel` working this task):

```python
    def fill(self, field_layout):
        """Interpolate every active monom and ``update`` it in the layout.

        TEMPORARY: kept only so the not-yet-updated NeuROMModel keeps working
        during the refactor. Task 4 removes this and routes interpolation through
        the injected IntegrationDomain instead.
        """
        for block in self._assemblies:
            for assembly in block:
                if not bool(assembly.active):
                    continue
                field_layout.update(assembly.field, assembly.interpolate())
```

- [ ] **Step 3: Replace `self._meshes[k]` with `axis.mesh` in `evaluate` and `assemble`**

In `evaluate`, the `PointWiseInterpolator` line currently reads:

```python
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
```

Replace with:

```python
                pwi = PointWiseInterpolator(
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
```

In `assemble`, the `PointWiseInterpolator` line currently reads:

```python
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
```

Replace with:

```python
                pwi = PointWiseInterpolator(
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
```

- [ ] **Step 4: Remove the now-unused `SeparatedDomain` import from `pgd.py`**

At the top of `src/neurom/decompositions/pgd.py`, delete the line:

```python
from neurom.interpolation.separated_domain import SeparatedDomain
```

(`Mesh` stays imported — it is now used only by `Axis.__post_init__`, which is fine.)

- [ ] **Step 5: Rewrite the affected unit tests in `test_pgd.py`**

Four groups of changes in `tests/unit/decompositions/test_pgd.py`:

**(a)** Replace `test_cppgd_owns_separated_domain_synced_with_truncation` entirely with:

```python
def test_n_modes_truncated_counts_active_blocks():
    model = CPPGD(axes=make_two_axes(), n_modes_max=3, n_modes_ini=1)
    assert model.n_modes_truncated == 1
    # exactly the leading block's assemblies are active
    assert all(bool(a.active) for a in model._assemblies[0])
    assert all(not bool(a.active) for a in model._assemblies[1])
    model.add_mode()
    assert model.n_modes_truncated == 2
    assert all(bool(a.active) for a in model._assemblies[1])


def test_assemblies_accessor_is_flat_and_shares_axis_contexts():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=3, n_modes_ini=1)
    flat = model.assemblies()
    assert len(flat) == 3 * 2                       # n_modes_max * n_axes
    # mode-major, axis order: block m, axis k -> flat[m * n_axes + k]
    assert flat[0].context is axes[0].context
    assert flat[1].context is axes[1].context
    assert flat[2].context is axes[0].context       # mode 1, axis 0


def test_no_requires_grad_param_in_inactive_assembly():
    """The illegal state (active=False, requires_grad=True) never occurs across
    the greedy sequence: every trainable monom belongs to an active assembly."""
    model = CPPGD(axes=make_two_axes(), n_modes_max=3, n_modes_ini=1)

    def check(mdl):
        for block in mdl._assemblies:
            if bool(block[0].active):
                continue
            for a in block:
                assert not a.field.values_reduced.requires_grad

    check(model)                 # initial
    model.freeze_mode(0)
    model.add_mode()             # mode 1 active, mode 0 frozen-but-active
    check(model)
    model.add_mode()             # capacity
    check(model)
```

**(b)** Replace the two `fill` tests. `test_fill_updates_active_monoms_matching_direct_assembly` uses `model._contexts[0]`; keep it but source the context from the axis (equivalent, public):

```python
def test_fill_updates_active_monoms_matching_direct_assembly():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=1)
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.ones_like(model.monoms[0][0].values_reduced)
        )
    layout = FieldLayout()
    model.register_into(layout)
    model.fill(layout)

    res = layout[model.monoms[0][0].name]
    expected = QuadratureAssembly(
        axes[0].context, axes[0].sf, model.monoms[0][0]
    ).interpolate()
    assert torch.allclose(res.u, expected.u)
```

`test_fill_leaves_inactive_monoms_uninterpolated` stays unchanged (still valid).

**(c)** Replace `model._meshes[...]` with `axes[...].mesh` in the three tests that use it: `test_assemble_matches_manual_outer_product`, `test_assemble_sums_two_modes_matching_manual_outer_products`, `test_evaluate_and_assemble_vector_factor`. For example, in `test_assemble_matches_manual_outer_product`:

```python
    pwi_s = PointWiseInterpolator(
        axes[0].mesh, axes[0].sf, model.monoms[0][0], axes[0].mapping
    )
    pwi_e = PointWiseInterpolator(
        axes[1].mesh, axes[1].sf, model.monoms[0][1], axes[1].mapping
    )
```

Apply the same `model._meshes[0] -> axes[0].mesh`, `model._meshes[1] -> axes[1].mesh` substitution in the other two tests (they bind `axes = make_two_axes()` / `[space, para]` already).

**(d)** Leave `test_add_mode_activates_new_without_freezing_previous`, `test_add_mode_raises_at_max`, `test_add_mode_to_optimizer_*`, `test_directory_*`, `test_evaluate_matched_pointwise_matches_assemble_diagonal`, `test_two_vector_axes_raises`, `test_register_into_*`, `test_cppgd_construction_structure_and_freeze` as they are — they do not touch removed internals.

- [ ] **Step 6: Run the CPPGD tests**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py -q`
Expected: PASS. The `NeuROMModel` tests in this file still use the 3-arg constructor and the temporary `fill` — they pass unchanged this task.

- [ ] **Step 7: Run the whole unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS — `test_separated_domain.py` still passes (`SeparatedDomain` still exists; deleted in Task 5).

- [ ] **Step 8: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "$(cat <<'EOF'
refactor(decompositions): CPPGD reads axis contexts, owns flagged assemblies

CPPGD no longer builds its own meshes/contexts or a SeparatedDomain: it
reads axis.context, owns mode-blocked QuadratureAssembly with `active`
flags, derives n_modes_truncated from those flags, and add_mode activates
before unfreezing. Adds assemblies() as the domain-building seam. fill()
kept temporarily until NeuROMModel takes an injected domain.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Decomposition stops interpolating — drop `fill`, inject domain into `NeuROMModel`

**Files:**
- Modify: `src/neurom/decompositions/base.py` (drop `fill` abstractmethod)
- Modify: `src/neurom/decompositions/pgd.py` (remove `CPPGD.fill`)
- Modify: `src/neurom/neurom_model.py` (take `integration_domain`; forward uses it)
- Test: `tests/unit/decompositions/test_pgd.py` (NeuROMModel tests + fake decomposition)

**Interfaces:**
- Consumes: `CPPGD.assemblies()` (Task 3); `IntegrationDomain` (Task 1).
- Produces:
  - `NeuROMModel(field_layout, decomposition, integration_domain, energy)` — 4 positional args, mirroring `FEMModel(mesh, field_layout, integration_domain, loss)`.
  - `NeuROMModel.forward()` (training) runs `self.integration_domain.interpolate_all(self.field_layout)` and returns the layout; `forward(coords)` (eval) returns `decomposition.evaluate(coords)`.
  - `TensorDecomposition` contract: `register_into`, `evaluate`, `assemble` (no `fill`).

- [ ] **Step 1: Write/adjust the failing tests**

In `tests/unit/decompositions/test_pgd.py`:

**(a)** The `_ConstantDecomposition` fake currently implements `fill`. It must instead register an assembly the caller can interpolate through a domain. Replace the class with:

```python
class _ConstantDecomposition(TensorDecomposition):
    """Minimal fake decomposition with NO CP structure: registers one fixed
    Field and exposes a QuadratureAssembly for it. Proves NeuROMModel is
    generic and interpolates through the injected IntegrationDomain."""

    def __init__(self):
        super().__init__()
        n = 4
        coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
        nodes = torch.arange(0, n)
        elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
        topo = Topology(nodes, elements)
        positions = Field(name="dummy_pos", topology=topo, values=coords)
        self.field = Field(name="dummy", topology=topo, values=torch.ones(n, 1))
        sf = LinearSegment()
        mesh = Mesh(topology=topo, nodes_positions=positions)
        ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(sf))
        self._assembly = QuadratureAssembly(ctx, sf, self.field)

    def register_into(self, field_layout):
        field_layout.add(self.field)

    def assemblies(self):
        return [self._assembly]

    def evaluate(self, coords):
        return torch.ones(coords[0].reshape(-1).shape[0], 1)

    def assemble(self, coords):
        return torch.ones(*[c.reshape(-1).shape[0] for c in coords])
```

Add the imports this fake now needs at the top of the test file if absent: `from neurom.meshes import Mesh` (currently only `Topology` is imported — change to `from neurom.meshes import Topology, Mesh`), `from neurom.interpolation.quadrature_context import QuadratureContext`, `from neurom.interpolation import IntegrationDomain`. `QuadratureAssembly`, `LinearSegment`, `TwoPoints1D`, `IsoparametricMapping1D` are already imported.

**(b)** Rewrite the four `NeuROMModel` tests to the 4-arg constructor + injected domain. Replace them with:

```python
def test_neurommodel_train_forward_returns_layout_and_optimizes():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    layout = FieldLayout()
    domain = IntegrationDomain(cppgd.assemblies())

    def energy(out):
        name = cppgd.directory()["space"][0]
        s = out[name]
        return integrate(s.u * s.measure)   # linear in S -> nonzero grad at 0 init

    model = NeuROMModel(layout, cppgd, domain, energy)
    out = model()                            # training forward
    assert out is layout                     # returns the filled layout

    before = cppgd.monoms[0][0].values_reduced.detach().clone()
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    optim.zero_grad()
    loss = model.energy(model())
    loss.backward()
    optim.step()
    after = cppgd.monoms[0][0].values_reduced.detach()
    assert not torch.allclose(before, after)


def test_neurommodel_eval_forward_matched_pointwise():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    with torch.no_grad():
        cppgd.monoms[0][0].values_reduced.copy_(torch.linspace(0.0, 4.0, 5).unsqueeze(-1))
        cppgd.monoms[0][1].values_reduced.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1))
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, energy=lambda out: out)
    model.eval()
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    pts = torch.stack([x, E], dim=1)
    u = model(pts)
    assert u.shape == (2, 1)
    assert torch.allclose(u, cppgd.evaluate(pts), atol=1e-6)


def test_neurommodel_eval_forward_requires_coords():
    cppgd = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, energy=lambda out: out)
    model.eval()
    with pytest.raises(ValueError):
        model()


def test_neurommodel_assemble_delegates():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, energy=lambda out: out)
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    assert torch.allclose(model.assemble([x, E]), cppgd.assemble([x, E]))


def test_neurommodel_is_format_agnostic():
    layout = FieldLayout()
    deco = _ConstantDecomposition()
    domain = IntegrationDomain(deco.assemblies())
    model = NeuROMModel(layout, deco, domain, energy=lambda out: out["dummy"].u.sum())
    out = model()                            # train: fills via the domain
    assert float(model.energy(out)) == out["dummy"].u.sum()
    model.eval()
    assert model([torch.zeros(3)]).shape == (3, 1)   # evaluate stub
```

(The old fake had a `.filled` flag; the new one is interpolated through the domain, so the assertion checks the layout was filled instead.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py -k neurommodel -v`
Expected: FAIL — `NeuROMModel.__init__` takes 3 args, not 4 (`TypeError`).

- [ ] **Step 3: Update `NeuROMModel`**

Replace `src/neurom/neurom_model.py` in full:

```python
import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class NeuROMModel(nn.Module):
    """Decomposition-driven model that reads like a classic ``nn.Module``.

    Counterpart of :class:`neurom.fem_model.FEMModel` for separated
    representations. Depends only on the :class:`TensorDecomposition` contract
    (``register_into`` / ``evaluate`` / ``assemble``), so any format (CP, later
    Tucker/TT) drives the same model. Interpolation is the injected
    :class:`~neurom.interpolation.integration_domain.IntegrationDomain`'s job —
    the same domain interpolates the decomposition's factor fields **and** any
    other field the energy reads (loads, sources).

    ``forward`` branches on ``self.training``:
      * training: interpolate every active field through the domain and **return
        the filled ``field_layout``** — the intermediate an external ``energy``
        consumes (``output = model(); loss = model.energy(output)``).
      * inference: ``forward(coords)`` returns the matched-pointwise field
        (``decomposition.evaluate(coords)``).

    Args:
        field_layout (FieldLayout): Fresh layout; ``__init__`` registers the
            decomposition's factor fields into it (a layout already holding those
            names raises ``ValueError`` on the duplicate registration).
        decomposition (TensorDecomposition): The separated representation.
        integration_domain (IntegrationDomain): Interpolates all active fields of
            the problem; typically ``IntegrationDomain([*decomposition.assemblies(),
            *other_assemblies])``.
        energy (Callable): Injected callable ``energy(output) -> torch.Tensor``
            (the counterpart of ``FEMModel.loss``), reading fields from the layout.
    """

    def __init__(self, field_layout, decomposition: TensorDecomposition,
                 integration_domain, energy):
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

    def assemble(self, coords):
        return self.decomposition.assemble(coords)
```

- [ ] **Step 4: Drop `fill` from the ABC**

In `src/neurom/decompositions/base.py`, delete the `fill` abstractmethod block:

```python
    @abstractmethod
    def fill(self, field_layout) -> None:
        """Interpolate the active factor fields and ``update()`` them in the layout.

        Called once per forward: the PGD analogue of
        :meth:`neurom.interpolation.integration_domain.IntegrationDomain.interpolate_all`.
        """
```

Update the class docstring line that mentions `fill`: change

```python
    relies on is: register those fields once (:meth:`register_into`) and
    re-interpolate the active ones per training forward (:meth:`fill`), then at
    inference evaluate the field pointwise (:meth:`evaluate`) or assemble the
    full grid (:meth:`assemble`). Format-specific structure readback and
    rank/mode enrichment stay on the concrete subclass.
```

to

```python
    relies on is: register those fields once (:meth:`register_into`), then at
    inference evaluate the field pointwise (:meth:`evaluate`) or assemble the
    full grid (:meth:`assemble`). Per-forward interpolation is the injected
    IntegrationDomain's job, not the decomposition's. Format-specific structure
    readback and rank/mode enrichment stay on the concrete subclass.
```

- [ ] **Step 5: Remove the temporary `CPPGD.fill`**

In `src/neurom/decompositions/pgd.py`, delete the entire temporary `fill` method added in Task 3 (the one whose docstring starts "TEMPORARY: kept only so the not-yet-updated NeuROMModel...").

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/decompositions/test_pgd.py -q`
Expected: PASS. Note the two `fill` tests from Task 3 — `test_fill_updates_active_monoms_matching_direct_assembly` and `test_fill_leaves_inactive_monoms_uninterpolated` — now call a removed method. **Delete both** in this step (their behaviour is now covered by `test_inactive_assembly_is_not_interpolated` in Task 1 and the domain-driven NeuROMModel tests). Re-run until green.

- [ ] **Step 7: Run the whole unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/neurom/decompositions/base.py src/neurom/decompositions/pgd.py \
        src/neurom/neurom_model.py tests/unit/decompositions/test_pgd.py
git commit -m "$(cat <<'EOF'
refactor: decomposition stops interpolating; NeuROMModel takes a domain

Drops fill() from the TensorDecomposition contract and from CPPGD.
NeuROMModel now takes an injected integration_domain (mirroring FEMModel)
and its training forward runs integration_domain.interpolate_all, so one
domain interpolates the monoms and any other field (loads) alike.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Delete `SeparatedDomain`

**Files:**
- Delete: `src/neurom/interpolation/separated_domain.py`
- Modify: `src/neurom/interpolation/__init__.py` (drop the export)
- Delete: `tests/unit/interpolation/test_separated_domain.py` (superseded by `test_integration_domain.py`)

**Interfaces:**
- Consumes: nothing new — Tasks 3 and 4 removed every internal reference.
- Produces: `neurom.interpolation` no longer exports `SeparatedDomain`.

- [ ] **Step 1: Confirm nothing references it anymore**

Run: `grep -rn "SeparatedDomain" src tests`
Expected: only `src/neurom/interpolation/separated_domain.py`, `src/neurom/interpolation/__init__.py`, and `tests/unit/interpolation/test_separated_domain.py` — no references in `pgd.py`, `neurom_model.py`, or `test_pgd.py`. If any other file matches, stop and fix it before deleting.

- [ ] **Step 2: Delete the source file and its test, drop the export**

```bash
git rm src/neurom/interpolation/separated_domain.py \
       tests/unit/interpolation/test_separated_domain.py
```

In `src/neurom/interpolation/__init__.py`, delete the line:

```python
from neurom.interpolation.separated_domain import SeparatedDomain
```

- [ ] **Step 3: Verify the import no longer resolves and the suite is green**

Run: `.venv/bin/python -c "import neurom.interpolation as i; assert not hasattr(i, 'SeparatedDomain'); print('ok')"`
Expected: `ok`

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(interpolation): remove SeparatedDomain

Truncation now lives in QuadratureAssembly.active + IntegrationDomain, and
CPPGD owns its assemblies directly, so the SeparatedDomain subclass and its
test are dead. Dropped the export.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire the load into the beam integration test through one domain

**Files:**
- Modify: `tests/integration/test_1d_beam_deflection_PGD_human.py`

**Interfaces:**
- Consumes: `CPPGD.assemblies()`, `IntegrationDomain`, `QuadratureAssembly`, the 4-arg `NeuROMModel`, `axis.context`.

This is the end-to-end validation of the whole change: the test currently
raises `RuntimeError: Field 'load' registered but not yet interpolated`; after
wiring it must run to completion with the rank-2 fit matching the analytical
target.

- [ ] **Step 1: Add the imports**

At the top of `tests/integration/test_1d_beam_deflection_PGD_human.py`, the file already imports `QuadratureContext`, `QuadratureAssembly`. Add:

```python
from neurom.interpolation.integration_domain import IntegrationDomain
```

- [ ] **Step 2: Build the load assembly and the shared domain, pass it to the model**

The test builds `axis_space`, `axis_E`, `pgd_approx = CPPGD(...)`, then `load_field = field_layout.add(Field(name="load", ...))`, then `model = NeuROMModel(field_layout=..., decomposition=pgd_approx, energy=...)`.

Replace the `NeuROMModel(...)` construction. Current:

```python
        # Creer le modele
        model = NeuROMModel(field_layout=field_layout,
                            decomposition=pgd_approx,
                            energy = lambda out: energy(out, pgd_approx, load_name="load"))
```

Replace with:

```python
        # One QuadratureAssembly for the load, sharing the SPACE axis context so
        # it is sampled at the same quadrature points as the space monoms.
        assembly_load = QuadratureAssembly(axis_space.context, sf, load_field)

        # ONE IntegrationDomain for the whole problem: the PGD monoms AND the
        # load. This is what fixes "Field 'load' registered but not interpolated".
        domain = IntegrationDomain([*pgd_approx.assemblies(), assembly_load])

        # Creer le modele
        model = NeuROMModel(field_layout=field_layout,
                            decomposition=pgd_approx,
                            integration_domain=domain,
                            energy=lambda out: energy(out, pgd_approx, load_name="load"))
```

The `energy(...)`, the training loop, and `add_mode` / `add_mode_to_optimizer` are untouched — the load is now interpolated, so `field_layout["load"]` resolves.

- [ ] **Step 3: Fix the plotting helper's private-mesh access**

The `factor()` helper inside `plot_solution` reads `pgd_approx._meshes[k]` (line ~208), which Task 3 removed. Change:

```python
        pwi = PointWiseInterpolator(
            pgd_approx._meshes[k],
            pgd_approx.axes[k].sf,
            pgd_approx.monoms[m][k],
            pgd_approx.axes[k].mapping,
        )
```

to use the now-public axis mesh:

```python
        pwi = PointWiseInterpolator(
            pgd_approx.axes[k].mesh,
            pgd_approx.axes[k].sf,
            pgd_approx.monoms[m][k],
            pgd_approx.axes[k].mapping,
        )
```

- [ ] **Step 4: Run the test to completion**

Run: `MPLBACKEND=Agg .venv/bin/python -c "import importlib.util; s=importlib.util.spec_from_file_location('t','tests/integration/test_1d_beam_deflection_PGD_human.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.Test1dBeamDeflection().test_beam(150)"`
Expected: prints `Successfully trained!` with no `RuntimeError`; writes `pgd_convergence.png` and `pgd_vs_analytical.png`. Inspect `pgd_vs_analytical.png` — the PGD sum-of-modes curve (panel 0/1) should track the analytical deflection.

- [ ] **Step 5: Run the full test suite**

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest tests -q`
Expected: PASS (unit + the non-PGD `test_1d_beam_deflection.py` integration test, which already used `IntegrationDomain([assembly_u, assembly_f])` and is unaffected).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_1d_beam_deflection_PGD_human.py
git commit -m "$(cat <<'EOF'
test(integration): interpolate the beam load through the shared domain

Builds one IntegrationDomain over the PGD monoms + the load assembly
(sharing the space axis context) and passes it to NeuROMModel. Fixes the
"Field 'load' registered but not yet interpolated" runtime error.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the entry on top**

Insert immediately under the header preamble (above the newest existing `## 2026-07-15` entry) in `CHANGELOG.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): one IntegrationDomain per problem, SeparatedDomain removed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** active flag (Task 1) · Axis context (Task 2) · CPPGD refactor + derived `n_modes_truncated` + `assemblies()` (Task 3) · `fill` off the contract + `NeuROMModel` domain injection (Task 4) · delete `SeparatedDomain` (Task 5) · load wiring end-to-end (Task 6) · forward-compat invariant "Axis builds context, never mapping" encoded in the Task 2 docstring · CHANGELOG (Task 7). The `active`/`requires_grad` invariant is enforced by `test_no_requires_grad_param_in_inactive_assembly` (Task 3).
- **Green at every boundary:** Task 3 keeps a temporary `CPPGD.fill` precisely so the suite stays green before Task 4 injects the domain; do not skip it.
- **Do not** change `src/neurom/decompositions/__init__.py` (already exports only `TensorDecomposition`, `Axis`, `CPPGD` — no `PGDFEMModel`, no `fill`).
- The deleted `tests/integration/test_1d_beam_deflection_PGD_test.py` (shown as `D` in git status) is a pre-existing removal unrelated to this plan; leave it deleted.
