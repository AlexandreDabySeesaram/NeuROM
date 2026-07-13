# NeuROMModel + SeparatedDomain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `PGDFEMModel`/`CPPGD.separated_view` with a decomposition-driven `NeuROMModel` (a classic-nn train/eval `nn.Module`), a truncation-aware `SeparatedDomain(IntegrationDomain)` that CPPGD fills through, and a `CPPGD.directory()` so a separable energy reads modes from the `FieldLayout` by name — with vector-ready `evaluate`/`assemble`.

**Architecture:** CPPGD owns a `SeparatedDomain` (one `QuadratureAssembly` per monom, built once, grouped by mode) that interpolates only active modes. `NeuROMModel` depends only on the extended `TensorDecomposition` ABC: training `forward()` fills the layout and returns it (energy applied outside); eval `forward(coords)` returns the matched-pointwise field. The separable energy stays a test-local function reading monoms via `directory()`.

**Tech Stack:** Python, PyTorch (`nn.Module`), pytest, `uv`.

## Global Constraints

- Run all Python via `uv run` — bare `python`/`pytest` are **not** on PATH (e.g. `uv run pytest`).
- Monom field names: `f"{self.name}_dim{axis.name}_mode{m}"` (e.g. name `"beam"`, axis `"space"`, mode 0 → `"beam_dimspace_mode0"`). `CPPGD.name` defaults to `"pgd"`.
- CP admits **at most one vector-valued factor per mode** — guarded in the constructor with `ValueError`.
- Keep the full suite green at every task boundary (`uv run pytest`). The final state must be green.
- Append a `CHANGELOG.md` entry (newest on top) before ending the session — done in Task 5.
- Follow existing library patterns: dependency injection, `nn.Module` submodules, `FieldLayout.add/update/__getitem__` contract, `QuadratureAssembly`/`PointWiseInterpolator` for interpolation.

Design spec: `docs/superpowers/specs/2026-07-13-neurom-model-separated-domain-design.md`

---

### Task 1: `SeparatedDomain(IntegrationDomain)`

Truncation-aware `IntegrationDomain`: assemblies grouped into mode-blocks; only the first `n_active_modes` blocks are interpolated; `grow()` activates the next.

**Files:**
- Create: `src/neurom/interpolation/separated_domain.py`
- Modify: `src/neurom/interpolation/__init__.py`
- Test: `tests/unit/interpolation/test_separated_domain.py`

**Interfaces:**
- Consumes: `IntegrationDomain` (`src/neurom/interpolation/integration_domain.py`), `QuadratureAssembly`, `FieldLayout`.
- Produces: `SeparatedDomain(mode_blocks: list[list[QuadratureAssembly]], n_active_modes: int)` with `.n_active_modes` (buffer), `.grow() -> int`, `.interpolate_all(field_layout) -> None`, inherited `.assemblies` / `._contexts` / `.update_contexts()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/interpolation/test_separated_domain.py`:

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
from neurom.interpolation import QuadratureContext, QuadratureAssembly, SeparatedDomain

torch.set_default_dtype(torch.float32)


def _blocks(n_modes=2, n=4):
    """Build `n_modes` mode-blocks, each one scalar 1-D monom on a shared context."""
    coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topo = Topology(nodes, elements)
    positions = Field(name="x", topology=topo, values=coords)
    sf = LinearSegment()
    mesh = Mesh(topology=topo, nodes_positions=positions)
    ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(sf))
    fields, blocks = [], []
    for m in range(n_modes):
        f = TrainableField(
            name=f"w{m}", topology=topo,
            init_values=torch.ones(n, 1), constraint=NoConstraint(),
        )
        fields.append(f)
        blocks.append([QuadratureAssembly(ctx, sf, f)])
    return blocks, fields, ctx


def _layout_with(fields):
    layout = FieldLayout()
    for f in fields:
        layout.add(f)
    return layout


def test_interpolate_all_only_active_blocks():
    blocks, fields, _ = _blocks(n_modes=2)
    domain = SeparatedDomain(blocks, n_active_modes=1)
    layout = _layout_with(fields)
    domain.interpolate_all(layout)
    assert layout[fields[0].name].u.shape[-1] == 1          # active
    with pytest.raises(RuntimeError):                        # inactive -> not interpolated
        _ = layout[fields[1].name]


def test_grow_activates_next_block_and_returns_index():
    blocks, fields, _ = _blocks(n_modes=2)
    domain = SeparatedDomain(blocks, n_active_modes=1)
    idx = domain.grow()
    assert idx == 1
    assert int(domain.n_active_modes) == 2
    layout = _layout_with(fields)
    domain.interpolate_all(layout)
    assert layout[fields[1].name].u.shape[-1] == 1          # now interpolated


def test_grow_raises_at_capacity():
    blocks, _, _ = _blocks(n_modes=1)
    domain = SeparatedDomain(blocks, n_active_modes=1)
    with pytest.raises(RuntimeError):
        domain.grow()


def test_contexts_deduplicated():
    blocks, _, ctx = _blocks(n_modes=3)                     # all share one context
    domain = SeparatedDomain(blocks, n_active_modes=3)
    assert len(domain._contexts) == 1
    assert domain._contexts[0] is ctx


def test_n_active_modes_is_a_buffer():
    blocks, _, _ = _blocks(n_modes=2)
    domain = SeparatedDomain(blocks, n_active_modes=1)
    assert "n_active_modes" in dict(domain.named_buffers())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/interpolation/test_separated_domain.py -v`
Expected: FAIL with `ImportError: cannot import name 'SeparatedDomain'`.

- [ ] **Step 3: Implement `SeparatedDomain`**

Create `src/neurom/interpolation/separated_domain.py`:

```python
import torch

from neurom.interpolation.integration_domain import IntegrationDomain


class SeparatedDomain(IntegrationDomain):
    """IntegrationDomain over mode-blocked assemblies; interpolates only active modes.

    Assemblies are grouped into blocks (one block per mode, each block one
    :class:`QuadratureAssembly` per axis). :meth:`interpolate_all` interpolates
    only the first ``n_active_modes`` blocks and ``update()``s them in the
    layout; :meth:`grow` activates the next block (greedy PGD enrichment). This
    is the truncation-aware analogue of :class:`IntegrationDomain`, which
    interpolates every assembly.

    Args:
        mode_blocks (list[list[QuadratureAssembly]]): One block per mode; each
            block holds one assembly per axis.
        n_active_modes (int): Number of initially active (interpolated) blocks.
    """

    def __init__(self, mode_blocks, n_active_modes):
        flat = [a for block in mode_blocks for a in block]
        super().__init__(flat)                     # dedups contexts, registers assemblies
        self._mode_blocks = mode_blocks            # same objects as self.assemblies
        self.register_buffer("n_active_modes", torch.tensor(int(n_active_modes)))

    def grow(self):
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

Modify `src/neurom/interpolation/__init__.py` — append:

```python
from neurom.interpolation.separated_domain import SeparatedDomain
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/interpolation/test_separated_domain.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: all previously-passing tests still pass (SeparatedDomain is purely additive).

- [ ] **Step 6: Commit**

```bash
git add src/neurom/interpolation/separated_domain.py src/neurom/interpolation/__init__.py tests/unit/interpolation/test_separated_domain.py
git commit -m "feat(interpolation): truncation-aware SeparatedDomain over mode-blocks"
```

---

### Task 2: CPPGD core refactor — `name`, own `SeparatedDomain`, `fill` via domain, `n_modes_truncated` property

Behaviour-preserving internal refactor. All existing CPPGD tests keep passing unchanged; two new tests pin the new structure. `separated_view` and `assemble` stay as-is (removed/changed in later tasks); `PGDFEMModel` untouched.

**Files:**
- Modify: `src/neurom/decompositions/pgd.py` (imports; `CPPGD.__init__`; add `n_modes_truncated` property; `add_mode`; `fill`)
- Test: `tests/unit/decompositions/test_pgd.py` (add two tests)

**Interfaces:**
- Consumes: `SeparatedDomain` (Task 1).
- Produces: `CPPGD(axes, n_modes_max, name="pgd", n_modes_ini=1)` with `.name`, `.domain: SeparatedDomain`, `.n_modes_truncated` (read-only int property → `domain.n_active_modes`), monoms named `f"{name}_dim{axis.name}_mode{m}"`. `add_mode()`/`fill()` unchanged in signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/decompositions/test_pgd.py`:

```python
def test_cppgd_has_name_and_monom_naming():
    model = CPPGD(axes=make_two_axes(), n_modes_max=2, n_modes_ini=1, name="beam")
    assert model.name == "beam"
    assert model.monoms[0][0].name == "beam_dimspace_mode0"
    assert model.monoms[1][1].name == "beam_dimE_mode1"


def test_cppgd_owns_separated_domain_synced_with_truncation():
    from neurom.interpolation import SeparatedDomain

    model = CPPGD(axes=make_two_axes(), n_modes_max=3, n_modes_ini=1)
    assert isinstance(model.domain, SeparatedDomain)
    assert int(model.domain.n_active_modes) == model.n_modes_truncated == 1
    model.add_mode()
    assert int(model.domain.n_active_modes) == model.n_modes_truncated == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_cppgd_has_name_and_monom_naming tests/unit/decompositions/test_pgd.py::test_cppgd_owns_separated_domain_synced_with_truncation -v`
Expected: FAIL — `CPPGD.__init__` has no `name` kwarg / no `.domain`.

- [ ] **Step 3: Refactor `CPPGD`**

In `src/neurom/decompositions/pgd.py`, add `SeparatedDomain` to the imports (after the `QuadratureAssembly` import):

```python
from neurom.interpolation.separated_domain import SeparatedDomain
```

Replace the entire `CPPGD.__init__` method with:

```python
    def __init__(self, axes, n_modes_max, name="pgd", n_modes_ini=1):
        super().__init__()
        self.name = name
        self.axes = list(axes)
        self.n_modes_max = n_modes_max

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

    @property
    def n_modes_truncated(self) -> int:
        """Number of currently-active modes (single source of truth: the domain)."""
        return int(self.domain.n_active_modes)
```

Replace the `add_mode` method with:

```python
    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Activates the next mode-block in the domain (zeroed out and trainable)
        without touching the freeze state of the currently-active modes. Returns
        the index of the newly-activated mode. Raises RuntimeError at capacity.
        """
        new = self.domain.grow()
        self._zero_out(new)
        self.unfreeze_mode(new)
        return new
```

Replace the `fill` method with:

```python
    def fill(self, field_layout):
        """Interpolate every active monom and ``update`` it in the layout.

        Delegates to the truncation-aware :class:`SeparatedDomain`: only active
        modes are interpolated. CP analogue of
        ``IntegrationDomain.interpolate_all``.
        """
        self.domain.interpolate_all(field_layout)
```

(`freeze_all`, `freeze_mode`, `unfreeze_mode`, `_zero_out`, `add_mode_to_optimizer`, `register_into`, `separated_view`, `assemble` are unchanged in this task.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS — the two new tests pass and every pre-existing CPPGD test still passes (the refactor is behaviour-preserving; `int(model.n_modes_truncated)` reads keep working via the property).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green (integration PGD test still uses `PGDFEMModel` + `separated_view`, both intact; `add_mode` still grows/zeros/unfreezes).

- [ ] **Step 6: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "refactor(decompositions): CPPGD name + SeparatedDomain, fill via domain, n_modes_truncated property"
```

---

### Task 3: CPPGD `directory()` + `evaluate()` + vector-ready `assemble()` + vector-factor guard

Additive API (`directory`, `evaluate`) plus a vector-ready rewrite of `assemble` that preserves scalar behaviour, plus a constructor guard for >1 vector axis.

**Files:**
- Modify: `src/neurom/decompositions/pgd.py` (constructor guard; add `directory`, `evaluate`; replace `assemble`)
- Test: `tests/unit/decompositions/test_pgd.py` (add `make_vector_axis` helper + 4 tests)

**Interfaces:**
- Consumes: `PointWiseInterpolator` (already imported in `pgd.py`), `string` (already imported).
- Produces: `CPPGD.directory() -> dict[str, list[str]]` (axis-major, active modes), `CPPGD.evaluate(coords) -> torch.Tensor` `(P, d)`, `CPPGD.assemble(coords) -> torch.Tensor` `(N_1,…,N_l[, d])`; constructor raises `ValueError` on >1 vector axis.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/decompositions/test_pgd.py` (the `make_vector_axis` helper and the tests):

```python
def make_vector_axis(name="space", n=5, lo=0.0, hi=10.0, dim=2):
    coords = torch.linspace(lo, hi, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(nodes, elements)
    positions = Field(name=f"{name}_positions", topology=topology, values=coords)
    sf = LinearSegment()
    return Axis(
        name=name, nodes_positions=positions, sf=sf,
        mapping=IsoparametricMapping1D(sf), quad=TwoPoints1D(),
        constraint=NoConstraint(), init_values=torch.zeros(n, dim),
    )


def test_directory_axis_major_active_names():
    model = CPPGD(axes=make_two_axes(), n_modes_max=3, n_modes_ini=2, name="beam")
    d = model.directory()
    assert set(d.keys()) == {"space", "E"}
    assert d["space"] == ["beam_dimspace_mode0", "beam_dimspace_mode1"]
    assert d["E"] == ["beam_dimE_mode0", "beam_dimE_mode1"]
    model.add_mode()
    assert model.directory()["space"] == [
        "beam_dimspace_mode0", "beam_dimspace_mode1", "beam_dimspace_mode2",
    ]


def test_evaluate_matched_pointwise_matches_assemble_diagonal():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(torch.linspace(0.0, 4.0, 5).unsqueeze(-1))
        model.monoms[0][1].values_reduced.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1))
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    u = model.evaluate([x, E])            # matched, (2, 1)
    grid = model.assemble([x, E])          # (2, 2)
    assert u.shape == (2, 1)
    assert torch.allclose(u.reshape(-1), torch.diagonal(grid), atol=1e-5)


def test_evaluate_and_assemble_vector_factor():
    space = make_vector_axis(name="space", n=5, dim=2)     # 2-D displacement factor
    para = make_axis(name="E", n=4, lo=100.0, hi=1000.0)   # scalar weight
    model = CPPGD(axes=[space, para], n_modes_max=1, n_modes_ini=1)
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(torch.arange(10, dtype=torch.float32).reshape(5, 2))
        model.monoms[0][1].values_reduced.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1))
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])

    u = model.evaluate([x, E])
    assert u.shape == (2, 2)               # (P, d)
    grid = model.assemble([x, E])
    assert grid.shape == (2, 2, 2)         # (N_x, N_E, d)

    from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
    pwi_s = PointWiseInterpolator(model._meshes[0], space.sf, model.monoms[0][0], space.mapping)
    pwi_g = PointWiseInterpolator(model._meshes[1], para.sf, model.monoms[0][1], para.mapping)
    S = pwi_s.at_position(x).reshape(2, 2)
    g = pwi_g.at_position(E).reshape(2, 1)
    assert torch.allclose(u, S * g, atol=1e-5)
    assert torch.allclose(torch.stack([grid[0, 0], grid[1, 1]]), u, atol=1e-5)


def test_two_vector_axes_raises():
    a1 = make_vector_axis(name="a", n=5, dim=2)
    a2 = make_vector_axis(name="b", n=4, dim=3)
    with pytest.raises(ValueError):
        CPPGD(axes=[a1, a2], n_modes_max=1, n_modes_ini=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -k "directory or evaluate or vector" -v`
Expected: FAIL — `directory`/`evaluate` don't exist; the vector `assemble` gives the wrong shape; no guard.

- [ ] **Step 3: Add the guard, `directory`, `evaluate`; replace `assemble`**

In `src/neurom/decompositions/pgd.py`, inside `CPPGD.__init__`, immediately after `self.axes = list(axes)` add the guard:

```python
        if sum(int(a.init_values.shape[1] > 1) for a in self.axes) > 1:
            raise ValueError(
                "CP-PGD admits at most one vector-valued factor per mode."
            )
```

Add two new methods (place after `register_into`, before `fill`):

```python
    def directory(self):
        """Ordered lookup table of active monom field names, keyed by axis.

        Returns:
            dict[str, list[str]]: axis name -> monom field names, one per active
            mode (index ``m``). Feed to a physics/energy term to read each monom
            out of the FieldLayout by name (``field_layout[name]``). Truncates to
            active modes; grows after :meth:`add_mode`.
        """
        n = self.n_modes_truncated
        return {
            axis.name: [self.monoms[m][k].name for m in range(n)]
            for k, axis in enumerate(self.axes)
        }

    def evaluate(self, coords):
        """Evaluate ``u`` at matched query points (diagonal), summed over modes.

        Args:
            coords (list[torch.Tensor]): one 1-D tensor per axis, all length ``P``.

        Returns:
            torch.Tensor: ``(P, d)`` = ``sum_m prod_k w_m^k(coords[k][p])``; ``d``
            is the single vector factor's dim, or 1 if all factors are scalar.
            Detached (via ``PointWiseInterpolator``).
        """
        n = self.n_modes_truncated
        total = None
        for m in range(n):
            prod = None
            for k, axis in enumerate(self.axes):
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1))   # (P, 1, dim_k)
                w = w.reshape(w.shape[0], -1)                # (P, dim_k)
                prod = w if prod is None else prod * w       # scalar * vector broadcasts
            total = prod if total is None else total + prod
        return total
```

Replace the existing `assemble` method (the last method in `CPPGD`) with the vector-ready version:

```python
    def assemble(self, coords):
        """Assemble the full separated tensor at the given per-axis coordinates.

        Args:
            coords (list[torch.Tensor]): One 1-D tensor per axis (length N_k),
                the query coordinates on that axis.

        Returns:
            torch.Tensor: full grid tensor of shape ``(N_1, ..., N_l[, d])``; the
            trailing ``d`` is present iff a vector factor exists (else dropped).
            Equals ``sum_m prod_k w_m^k`` over the coordinate grid. Detached.
        """
        n_modes = self.n_modes_truncated
        mode_letter = "Z"
        per_axis = []  # per_axis[k]: (n_modes, N_k) or (n_modes, N_k, d_k)
        for k, axis in enumerate(self.axes):
            P_k = coords[k].reshape(-1).shape[0]
            cols = []
            for m in range(n_modes):
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1)).reshape(P_k, -1)  # (N_k, d_k)
                cols.append(w.reshape(-1) if w.shape[1] == 1 else w)
            per_axis.append(torch.stack(cols, dim=0))

        grid_letters = string.ascii_lowercase[: len(self.axes)]
        comp_pool = iter(c for c in string.ascii_uppercase if c != mode_letter)
        in_subs, out_grid, out_comp = [], "", ""
        for k, arr in enumerate(per_axis):
            sub = mode_letter + grid_letters[k]
            out_grid += grid_letters[k]
            if arr.dim() == 3:  # vector axis: (n_modes, N_k, d_k)
                c = next(comp_pool)
                sub += c
                out_comp += c
            in_subs.append(sub)
        return torch.einsum(f"{','.join(in_subs)}->{out_grid}{out_comp}", *per_axis)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS — the 4 new tests pass; the pre-existing scalar `assemble` tests (`test_assemble_matches_manual_outer_product`, `test_assemble_sums_two_modes_matching_manual_outer_products`) still pass with output shape `(2, 2)` (scalar → no component index).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): CPPGD.directory + vector-ready evaluate/assemble + one-vector-factor guard"
```

---

### Task 4: `TensorDecomposition` ABC extension + `NeuROMModel` (coexisting with `PGDFEMModel`)

Extend the ABC with `evaluate`/`assemble`; add the top-level `NeuROMModel`; update the `_ConstantDecomposition` fake to satisfy the ABC. `PGDFEMModel`, `separated_view`, and their tests stay green in this task (removed in Task 5).

**Files:**
- Modify: `src/neurom/decompositions/base.py` (add abstract `evaluate`, `assemble`)
- Create: `src/neurom/neurom_model.py`
- Modify: `tests/unit/decompositions/test_pgd.py` (update `_ConstantDecomposition`; add `NeuROMModel` import + tests)

**Interfaces:**
- Consumes: extended `TensorDecomposition` (`register_into`, `fill`, `evaluate`, `assemble`); `CPPGD.directory`/`evaluate`/`assemble` (Task 3).
- Produces: `NeuROMModel(field_layout, decomposition, energy)` with `forward(coords=None)` (train → returns filled `field_layout`; eval → `decomposition.evaluate(coords)`), `.energy` attribute, `.assemble(coords)`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/decompositions/test_pgd.py`, add the import near the other imports:

```python
from neurom.neurom_model import NeuROMModel
```

Add `evaluate`/`assemble` to the existing `_ConstantDecomposition` class body (it must satisfy the now-abstract ABC methods):

```python
    def evaluate(self, coords):
        return torch.ones(coords[0].reshape(-1).shape[0], 1)

    def assemble(self, coords):
        return torch.ones(*[c.reshape(-1).shape[0] for c in coords])
```

Append these tests:

```python
def test_neurommodel_train_forward_returns_layout_and_optimizes():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    layout = FieldLayout()

    def energy(out):
        name = cppgd.directory()["space"][0]
        s = out[name]
        return integrate(s.u * s.measure)   # linear in S -> nonzero grad at 0 init

    model = NeuROMModel(layout, cppgd, energy)
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
    model = NeuROMModel(FieldLayout(), cppgd, energy=lambda out: out)
    model.eval()
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    u = model([x, E])
    assert u.shape == (2, 1)
    assert torch.allclose(u, cppgd.evaluate([x, E]), atol=1e-6)


def test_neurommodel_eval_forward_requires_coords():
    cppgd = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    model = NeuROMModel(FieldLayout(), cppgd, energy=lambda out: out)
    model.eval()
    with pytest.raises(ValueError):
        model()


def test_neurommodel_assemble_delegates():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    model = NeuROMModel(FieldLayout(), cppgd, energy=lambda out: out)
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    assert torch.allclose(model.assemble([x, E]), cppgd.assemble([x, E]))


def test_neurommodel_is_format_agnostic():
    layout = FieldLayout()
    deco = _ConstantDecomposition()
    model = NeuROMModel(layout, deco, energy=lambda out: out["dummy"].u.sum())
    out = model()                            # train: fills
    assert deco.filled
    assert float(model.energy(out)) == 1.0
    model.eval()
    assert model([torch.zeros(3)]).shape == (3, 1)   # evaluate stub
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -k neurommodel -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError: neurom.neurom_model` (and the ABC/fake are not yet aligned).

- [ ] **Step 3: Extend the ABC and implement `NeuROMModel`**

In `src/neurom/decompositions/base.py`, add two abstract methods after `fill` (inside the class):

```python
    @abstractmethod
    def evaluate(self, coords):
        """Evaluate the field at matched query points (inference). Returns ``(P, d)``."""

    @abstractmethod
    def assemble(self, coords):
        """Assemble the full grid tensor over the given per-axis coordinates."""
```

Create `src/neurom/neurom_model.py`:

```python
import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class NeuROMModel(nn.Module):
    """Decomposition-driven model that reads like a classic ``nn.Module``.

    Counterpart of :class:`neurom.fem_model.FEMModel` for separated
    representations. Depends only on the :class:`TensorDecomposition` contract,
    so any format (CP, later Tucker/TT) drives the same model.

    ``forward`` branches on ``self.training``:
      * training: fill the layout via the decomposition and **return the filled
        ``field_layout``** — the intermediate output an external ``energy``
        consumes (``output = model(); loss = model.energy(output)``).
      * inference: ``forward(coords)`` returns the matched-pointwise field
        (``decomposition.evaluate(coords)``).

    Args:
        field_layout (FieldLayout): Fresh layout; ``__init__`` registers the
            decomposition's factor fields into it (a layout already holding those
            names raises ``ValueError`` on the duplicate registration).
        decomposition (TensorDecomposition): The separated representation.
        energy (Callable): Injected callable ``energy(output) -> torch.Tensor``
            (the counterpart of ``FEMModel.loss``), reading modes from the layout.
    """

    def __init__(self, field_layout, decomposition: TensorDecomposition, energy):
        super().__init__()
        self.field_layout = field_layout
        self.decomposition = decomposition
        self.energy = energy
        decomposition.register_into(field_layout)

    def forward(self, coords=None):
        if self.training:
            self.decomposition.fill(self.field_layout)
            return self.field_layout
        if coords is None:
            raise ValueError(
                "eval forward requires coords (one 1-D tensor per axis, matched length)."
            )
        return self.decomposition.evaluate(coords)

    def assemble(self, coords):
        return self.decomposition.assemble(coords)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS — new `NeuROMModel` tests pass; the updated `_ConstantDecomposition` instantiates (implements all four abstract methods); existing `PGDFEMModel` tests still pass (`PGDFEMModel` does not call `evaluate`/`assemble`).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: green (`CPPGD` implements `evaluate`/`assemble`; the integration test still uses `PGDFEMModel`).

- [ ] **Step 6: Commit**

```bash
git add src/neurom/decompositions/base.py src/neurom/neurom_model.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): NeuROMModel + evaluate/assemble on TensorDecomposition ABC"
```

---

### Task 5: Migrate to `NeuROMModel`/`directory`, remove `PGDFEMModel` + `separated_view`

Migrate the integration test onto the new flow, delete the superseded API, and record the change.

**Files:**
- Modify: `tests/integration/test_1d_beam_deflection_PGD_test.py` (energy via `directory`; `NeuROMModel` wiring; eval-mode assertion)
- Modify: `tests/unit/decompositions/test_pgd.py` (remove `separated_view`/`PGDFEMModel` tests + import; add removal-regression tests)
- Modify: `src/neurom/decompositions/pgd.py` (remove `separated_view`)
- Delete: `src/neurom/decompositions/pgd_fem_model.py`
- Modify: `src/neurom/decompositions/__init__.py` (drop `PGDFEMModel`)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `NeuROMModel` (Task 4), `CPPGD.directory` (Task 3).
- Produces: final green suite; `neurom.decompositions` exports `TensorDecomposition`, `Axis`, `CPPGD` only.

- [ ] **Step 1: Migrate the integration test**

In `tests/integration/test_1d_beam_deflection_PGD_test.py`:

Change the decomposition import line (`from neurom.decompositions import Axis, CPPGD, PGDFEMModel`) to:

```python
from neurom.decompositions import Axis, CPPGD
from neurom.neurom_model import NeuROMModel
```

Replace the body of `potential_energy` — its first block that calls `separated_view` — so it reads via `directory()`:

```python
def potential_energy(cppgd, field_layout, f_value):
    """External separable parametric energy, read from the filled FieldLayout via directory()."""
    d = cppgd.directory()
    space = [field_layout[name] for name in d["space"]]
    para = [field_layout[name] for name in d["E"]]
    n_modes = len(space)

    dS = [jacobian_field(space[m].x, space[m].u).reshape(space[m].u.shape)
          for m in range(n_modes)]         # each (N_e, N_q, 1)
    S = [space[m].u for m in range(n_modes)]
    measure_s = space[0].measure

    g = [para[m].u for m in range(n_modes)]
    E_val = para[0].x                       # coordinate on the E axis == E
    measure_E = para[0].measure

    elastic = 0.0
    for m in range(n_modes):
        for n in range(n_modes):
            Kx = integrate(dS[m] * dS[n] * measure_s)
            AE = integrate(E_val * g[m] * g[n] * measure_E)
            elastic = elastic + Kx * AE
    elastic = 0.5 * elastic

    load = 0.0
    for m in range(n_modes):
        Fx = integrate(f_value * S[m] * measure_s)
        Gm = integrate(g[m] * measure_E)
        load = load + Fx * Gm

    return elastic + load
```

In **both** test methods, replace the `cppgd = CPPGD(...)` + `PGDFEMModel(...)` wiring with the block below. Use `n_modes_max=1` in `test_parametric_beam_matches_analytical` and `n_modes_max=2` in `test_greedy_enrichment_second_mode_stays_bounded` (matching the originals — that is the only difference between the two):

```python
        cppgd = CPPGD(axes=[space_axis, para_axis], n_modes_max=1, n_modes_ini=1, name="beam")
        field_layout = FieldLayout()
        model = NeuROMModel(
            field_layout, cppgd,
            energy=lambda out: potential_energy(cppgd, out, f_value),
        )
```

In **both** `closure`/`closure2` bodies, compute the loss through the model's energy:

```python
        def closure():
            optimizer.zero_grad()
            out = model()
            loss = model.energy(out)
            loss.backward(retain_graph=True)
            return loss
```

(and likewise `closure2` with `optimizer2`).

In **both** tests, change the final assembled comparison from `u = cppgd.assemble([x_test, E_test])` to `u = model.assemble([x_test, E_test])`.

In `test_parametric_beam_matches_analytical`, after the assembled-tensor assertion, add an eval-mode matched-pointwise check:

```python
        model.eval()
        x_pts = torch.linspace(x_min, x_max, 7)
        E_pts = torch.linspace(E_min, E_max, 7)
        u_pw = model([x_pts, E_pts])                       # matched pointwise, (7, 1)
        u_pw_analytical = 0.5 * f_value * (x_pts - x_min) * (x_pts - x_max) / E_pts
        scale_pw = float(u_pw_analytical.abs().max())
        assert u_pw.reshape(-1).numpy() == pytest.approx(
            u_pw_analytical.numpy(), abs=self.relative_tolerance * scale_pw
        )
```

- [ ] **Step 2: Run the integration test (new API works, `separated_view` still present but unused)**

Run: `uv run pytest tests/integration/test_1d_beam_deflection_PGD_test.py -v`
Expected: PASS (2 passed) — both solves match the analytical field, and the new eval-mode pointwise check passes.

- [ ] **Step 3: Prune superseded unit tests + add removal regressions**

In `tests/unit/decompositions/test_pgd.py`:
- Remove the import line `from neurom.decompositions import PGDFEMModel`.
- Delete these tests: `test_separated_view_keys_shapes_and_values`, `test_separated_view_reflects_added_mode`, `test_pgdfemmodel_forward_returns_scalar_and_optimizes`, `test_pgdfemmodel_is_format_agnostic`.
- Keep `_ConstantDecomposition` (now used by `test_neurommodel_is_format_agnostic`) and `test_interpolate_separated_is_removed`.
- Add removal-regression tests:

```python
def test_separated_view_is_removed():
    model = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    assert not hasattr(model, "separated_view")


def test_pgdfemmodel_export_is_removed():
    import neurom.decompositions as d
    assert not hasattr(d, "PGDFEMModel")
```

- [ ] **Step 4: Run the unit tests (still green with `separated_view` present)**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: `test_separated_view_is_removed` and `test_pgdfemmodel_export_is_removed` FAIL (still present); everything else passes. This pins the removals for Step 5.

- [ ] **Step 5: Remove `separated_view`, delete `PGDFEMModel`, update exports**

In `src/neurom/decompositions/pgd.py`, delete the entire `separated_view` method.

Delete the file `src/neurom/decompositions/pgd_fem_model.py`:

```bash
git rm src/neurom/decompositions/pgd_fem_model.py
```

In `src/neurom/decompositions/__init__.py`, remove the line `from neurom.decompositions.pgd_fem_model import PGDFEMModel`. Final contents:

```python
from neurom.decompositions.base import TensorDecomposition
from neurom.decompositions.pgd import Axis, CPPGD
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: green — including `test_separated_view_is_removed` and `test_pgdfemmodel_export_is_removed`.

- [ ] **Step 7: Update the CHANGELOG**

Prepend to `CHANGELOG.md` (newest on top):

```markdown
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
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(decompositions): migrate to NeuROMModel/directory, remove PGDFEMModel + separated_view"
```

---

## Notes for the implementer

- **Green at every boundary.** Tasks 1–4 are additive/behaviour-preserving; only Task 5 removes API, and it sequences the integration migration (Step 1) before the deletions (Step 5) so nothing is red across a commit except the intended removal-regression tests inside Task 5.
- **Param dedup:** monom `TrainableField`s are referenced by `CPPGD.monoms`, the domain's assemblies, and (after `register_into`) the `FieldLayout`. `nn.Module.parameters()` dedups by identity — do not "fix" the apparent duplication.
- **`n_modes_truncated` is read-only** now (a property). The only mutators of the active count are the constructor and `add_mode` (via `domain.grow()`).
- **Scalar `assemble` output is unchanged** (`(N_1,…,N_l)`); the component index only appears when a factor is vector-valued.
