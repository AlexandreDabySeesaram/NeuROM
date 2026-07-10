# CP-PGD on FieldLayout / FEMModel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CPPGD` fill a caller-supplied `FieldLayout` and be driven by a new, format-agnostic `PGDFEMModel`, so PGD lives inside the library's dependency-injected pipeline instead of being a parallel island.

**Architecture:** Introduce a `TensorDecomposition` ABC (nn.Module + ABC) with exactly two abstract methods — `register_into(field_layout)` and `fill(field_layout)`. `CPPGD` implements them (plus a `separated_view` readback), replacing the old `interpolate_separated()`. `PGDFEMModel(decomposition, field_layout, loss)` depends only on the ABC. The separable energy stays an external function, refactored to read from the filled layout via `separated_view`.

**Tech Stack:** Python, PyTorch (`torch`, `torch.nn`), pytest, `uv` (run everything with `uv run`).

## Global Constraints

- Run all Python via `uv run` — bare `python`/`pytest` are not on PATH.
- `torch.set_default_dtype(torch.float32)` at the top of every test module (existing convention).
- **No changes to `src/neurom/field_layout.py`.** CP-PGD must use its existing `add(field)` / `update(field, result)` / `layout[name]` contract unchanged.
- Package dependency direction is one-way: `neurom.decompositions` may import `neurom` core, never the reverse. Keep `PGDFEMModel` inside `decompositions/`.
- Monom field naming is unchanged: `f"{axis.name}_mode{m}"` (e.g. `"space_mode0"`). These are the flat layout keys.
- Follow CLAUDE.md: append a CHANGELOG entry (newest on top) before ending the session.
- After each task the full suite must stay green: `uv run pytest`.

---

### Task 1: `TensorDecomposition` ABC + `CPPGD.register_into` / `CPPGD.fill`

Introduce the abstraction seam and give `CPPGD` the two layout methods. The ABC has two abstract methods, so `CPPGD` must implement both in this task to remain instantiable (existing tests instantiate it).

**Files:**
- Create: `src/neurom/decompositions/base.py`
- Modify: `src/neurom/decompositions/pgd.py` (class declaration + imports + two new methods)
- Modify: `src/neurom/decompositions/__init__.py` (export `TensorDecomposition`)
- Test: `tests/unit/decompositions/test_pgd.py` (append new tests + imports)

**Interfaces:**
- Consumes (existing, unchanged): `CPPGD(axes, n_modes_max, n_modes_ini=1)`; `CPPGD.monoms` (`ModuleList[ModuleList[TrainableField]]`); `CPPGD.axes` (`list[Axis]`); `CPPGD._contexts` (`ModuleList[QuadratureContext]`); `CPPGD.n_modes_truncated` (int buffer); `Axis.sf`; `TrainableField.name`; `FieldLayout.add(field)`, `FieldLayout.update(field, result)`, `FieldLayout[name]`; `QuadratureAssembly(context, sf, field).interpolate() -> QuadratureAssemblyResult`.
- Produces: `TensorDecomposition` (ABC, abstract `register_into(self, field_layout)`, `fill(self, field_layout)`); `CPPGD.register_into(field_layout) -> None`; `CPPGD.fill(field_layout) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/decompositions/test_pgd.py`. First extend the imports at the top of the file (add these lines beneath the existing imports):

```python
from neurom.decompositions import TensorDecomposition
from neurom.field_layout import FieldLayout
```

Then append these tests at the end of the file:

```python
def test_cppgd_is_a_tensor_decomposition():
    model = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    assert isinstance(model, TensorDecomposition)


def test_register_into_populates_layout_with_all_monoms():
    model = CPPGD(axes=make_two_axes(), n_modes_max=3, n_modes_ini=1)
    layout = FieldLayout()
    model.register_into(layout)
    # every monom name registered (3 modes x 2 axes), incl. inactive modes
    for mode in model.monoms:
        for f in mode:
            assert f.name in layout._fields


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
    ctx = model._contexts[0]
    expected = QuadratureAssembly(ctx, axes[0].sf, model.monoms[0][0]).interpolate()
    assert torch.allclose(res.u, expected.u)


def test_fill_leaves_inactive_monoms_uninterpolated():
    model = CPPGD(axes=make_two_axes(), n_modes_max=2, n_modes_ini=1)
    layout = FieldLayout()
    model.register_into(layout)
    model.fill(layout)
    # mode 1 inactive: registered but never interpolated -> RuntimeError on read
    with pytest.raises(RuntimeError):
        _ = layout[model.monoms[1][0].name]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -k "tensor_decomposition or register_into or fill_updates or fill_leaves" -v`
Expected: FAIL — `ImportError: cannot import name 'TensorDecomposition'` (and, once that is fixed, `AttributeError: 'CPPGD' object has no attribute 'register_into'`).

- [ ] **Step 3: Create the `TensorDecomposition` ABC**

Create `src/neurom/decompositions/base.py`:

```python
from abc import ABC, abstractmethod

import torch.nn as nn


class TensorDecomposition(nn.Module, ABC):
    """A tensor/separated decomposition that can populate a FieldLayout.

    Concrete formats (CP, and later Tucker / TT) own their own factor fields and
    know how many they are. The only contract :class:`PGDFEMModel` relies on is:
    register those fields once (:meth:`register_into`), then re-interpolate the
    active ones per forward (:meth:`fill`). Format-specific structure readback,
    assembly and rank/mode enrichment stay on the concrete subclass.
    """

    @abstractmethod
    def register_into(self, field_layout) -> None:
        """Register this decomposition's factor fields in the layout (setup)."""

    @abstractmethod
    def fill(self, field_layout) -> None:
        """Interpolate the active factor fields and ``update()`` them in the layout.

        Called once per forward: the PGD analogue of
        :meth:`neurom.interpolation.integration_domain.IntegrationDomain.interpolate_all`.
        """
```

- [ ] **Step 4: Make `CPPGD` implement the contract**

In `src/neurom/decompositions/pgd.py`, add the base import beneath the existing imports (after line 16, `from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator`):

```python
from neurom.decompositions.base import TensorDecomposition
```

Change the class declaration from `class CPPGD(nn.Module):` to:

```python
class CPPGD(TensorDecomposition):
```

Add these two methods to `CPPGD` (place them just above `interpolate_separated`):

```python
    def register_into(self, field_layout):
        """Register every monom field (all modes, all axes) in the layout.

        Called once at setup. All ``n_modes_max`` modes are registered up front
        (including not-yet-active ones) so ``add_mode`` needs no layout
        reference. Inactive monoms are registered but never interpolated.
        """
        for mode in self.monoms:
            for field in mode:
                field_layout.add(field)

    def fill(self, field_layout):
        """Interpolate every active monom and ``update`` it in the layout.

        CP analogue of ``IntegrationDomain.interpolate_all``: for each active
        mode and each axis, interpolate the monom at that axis's quadrature
        points and store the result under the monom's name in the layout.
        """
        for m in range(int(self.n_modes_truncated)):
            for k, axis in enumerate(self.axes):
                assembly = QuadratureAssembly(
                    self._contexts[k], axis.sf, self.monoms[m][k]
                )
                field_layout.update(self.monoms[m][k], assembly.interpolate())
```

Update `src/neurom/decompositions/__init__.py` to:

```python
from neurom.decompositions.base import TensorDecomposition
from neurom.decompositions.pgd import Axis, CPPGD
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (the four new tests plus all pre-existing ones — `interpolate_separated` is still present at this task).

- [ ] **Step 6: Commit**

```bash
git add src/neurom/decompositions/base.py src/neurom/decompositions/pgd.py src/neurom/decompositions/__init__.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): TensorDecomposition ABC + CPPGD register_into/fill"
```

---

### Task 2: `CPPGD.separated_view` + drop `interpolate_separated`

Add the layout readback and make the layout flow canonical by removing the old self-contained view. Migrate the one unit test that used it.

**Files:**
- Modify: `src/neurom/decompositions/pgd.py` (add `separated_view`, remove `interpolate_separated`)
- Test: `tests/unit/decompositions/test_pgd.py` (replace one test, add two)

**Interfaces:**
- Consumes: `CPPGD.register_into`, `CPPGD.fill` (Task 1); `CPPGD.monoms`, `CPPGD.axes`, `CPPGD.n_modes_truncated`; `FieldLayout[name] -> QuadratureAssemblyResult`.
- Produces: `CPPGD.separated_view(field_layout) -> dict[str, list[QuadratureAssemblyResult]]` (axis name -> list indexed by active mode). `CPPGD.interpolate_separated` no longer exists.

- [ ] **Step 1: Write/replace the failing tests**

In `tests/unit/decompositions/test_pgd.py`, **replace** the whole existing `test_interpolate_separated_keys_shapes_and_values` function (lines beginning `def test_interpolate_separated_keys_shapes_and_values():`) with:

```python
def test_separated_view_keys_shapes_and_values():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=2)

    # Give mode-0 space monom known nodal values so we can predict the result.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.ones_like(model.monoms[0][0].values_reduced)
        )

    layout = FieldLayout()
    model.register_into(layout)
    model.fill(layout)
    sep = model.separated_view(layout)

    # dict keyed by axis names, each a list over active modes
    assert set(sep.keys()) == {"space", "E"}
    assert len(sep["space"]) == 2 and len(sep["E"]) == 2

    res = sep["space"][0]
    # (N_e, N_q, u_dim): space mesh has 4 elements, TwoPoints1D -> 2 points, dim 1
    assert res.u.shape == (4, 2, 1)
    assert res.x.shape == (4, 2, 1)
    assert res.measure.shape == (4, 2, 1)

    # Ground truth: reference assembly of the same monom on the same context.
    ctx = model._contexts[0]
    expected = QuadratureAssembly(ctx, axes[0].sf, model.monoms[0][0]).interpolate()
    assert torch.allclose(res.u, expected.u)


def test_separated_view_reflects_added_mode():
    model = CPPGD(axes=make_two_axes(), n_modes_max=2, n_modes_ini=1)
    layout = FieldLayout()
    model.register_into(layout)
    model.fill(layout)
    assert len(model.separated_view(layout)["space"]) == 1

    model.add_mode()
    model.fill(layout)
    assert len(model.separated_view(layout)["space"]) == 2


def test_interpolate_separated_is_removed():
    model = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    assert not hasattr(model, "interpolate_separated")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -k "separated_view or interpolate_separated_is_removed" -v`
Expected: FAIL — `AttributeError: 'CPPGD' object has no attribute 'separated_view'`.

- [ ] **Step 3: Add `separated_view` and remove `interpolate_separated`**

In `src/neurom/decompositions/pgd.py`, **delete** the entire `interpolate_separated` method (the block starting `def interpolate_separated(self):` through its `return result`) and **replace** it with:

```python
    def separated_view(self, field_layout):
        """Read the active monoms' interpolations back out of the layout.

        The layout must have been ``fill``ed first (guaranteed inside
        ``PGDFEMModel.forward``, which fills before calling the loss).

        Returns:
            dict[str, list[QuadratureAssemblyResult]]: axis name -> list indexed
            by active mode; entry ``m`` is the interpolation of the single monom
            ``w_m^axis``. Enables writing separable energies monom by monom.
        """
        result = {}
        for k, axis in enumerate(self.axes):
            result[axis.name] = [
                field_layout[self.monoms[m][k].name]
                for m in range(int(self.n_modes_truncated))
            ]
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (all unit tests, including the three new/replaced ones).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): CPPGD.separated_view replaces interpolate_separated"
```

---

### Task 3: `PGDFEMModel` (format-agnostic) + wiring/agnosticism tests

Add the PGD FEM model that depends only on `TensorDecomposition`, and prove with a fake decomposition that it carries no CP dependency.

**Files:**
- Create: `src/neurom/decompositions/pgd_fem_model.py`
- Modify: `src/neurom/decompositions/__init__.py` (export `PGDFEMModel`)
- Test: `tests/unit/decompositions/test_pgd.py` (append imports + tests)

**Interfaces:**
- Consumes: `TensorDecomposition.register_into`, `TensorDecomposition.fill` (Tasks 1); `CPPGD.separated_view` (Task 2); `FieldLayout`; `integrate(tensor) -> scalar`; `QuadratureAssemblyResult(x, u, measure)`; `Field(name, topology, values)`; `Topology(nodes, elements)`.
- Produces: `PGDFEMModel(decomposition: TensorDecomposition, field_layout, loss)`; `PGDFEMModel.forward() -> torch.Tensor` (`decomposition.fill(layout)` then `loss()`); constructor calls `decomposition.register_into(field_layout)`.

- [ ] **Step 1: Write the failing tests**

Append to the imports block of `tests/unit/decompositions/test_pgd.py`:

```python
from neurom.decompositions import PGDFEMModel
from neurom.integrate import integrate
from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult
```

Append these tests at the end of the file:

```python
def test_pgdfemmodel_forward_returns_scalar_and_optimizes():
    axes = make_two_axes()
    cppgd = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    layout = FieldLayout()

    # Linear-in-S loss: gradient is the (nonzero) shape-function * measure, so
    # even a zero-initialised monom gets a nonzero update.
    def loss():
        s = cppgd.separated_view(layout)["space"][0]
        return integrate(s.u * s.measure)

    model = PGDFEMModel(cppgd, layout, loss)

    out = model()
    assert out.ndim == 0  # scalar

    before = cppgd.monoms[0][0].values_reduced.detach().clone()
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    optim.zero_grad()
    model().backward()
    optim.step()
    after = cppgd.monoms[0][0].values_reduced.detach()
    assert not torch.allclose(before, after)


class _ConstantDecomposition(TensorDecomposition):
    """Minimal fake decomposition with NO CP structure: registers one fixed
    Field and fills it with a constant result. Proves PGDFEMModel is generic."""

    def __init__(self):
        super().__init__()
        n = 3
        nodes = torch.arange(0, n)
        elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
        topo = Topology(nodes, elements)
        self.field = Field(name="dummy", topology=topo, values=torch.zeros(n, 1))
        self.filled = False

    def register_into(self, field_layout):
        field_layout.add(self.field)

    def fill(self, field_layout):
        self.filled = True
        res = QuadratureAssemblyResult(
            x=torch.zeros(1, 1, 1), u=torch.ones(1, 1, 1), measure=torch.ones(1, 1, 1)
        )
        field_layout.update(self.field, res)


def test_pgdfemmodel_is_format_agnostic():
    layout = FieldLayout()
    deco = _ConstantDecomposition()

    def loss():
        return layout["dummy"].u.sum()

    model = PGDFEMModel(deco, layout, loss)
    out = model()
    assert deco.filled
    assert float(out) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -k "pgdfemmodel" -v`
Expected: FAIL — `ImportError: cannot import name 'PGDFEMModel'`.

- [ ] **Step 3: Create `PGDFEMModel`**

Create `src/neurom/decompositions/pgd_fem_model.py`:

```python
import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class PGDFEMModel(nn.Module):
    """FEM model driving a tensor decomposition through a FieldLayout.

    PGD analogue of :class:`neurom.fem_model.FEMModel`. Depends only on the
    :class:`TensorDecomposition` contract, so any format (CP, and later Tucker /
    TT) drives the same model. Registers the decomposition's factor fields into
    the layout at construction; each forward re-interpolates the active ones and
    evaluates the (external) loss.

    Args:
        decomposition (TensorDecomposition): The separated representation.
        field_layout (FieldLayout): Flat layout the decomposition fills.
        loss (Callable[[], torch.Tensor]): No-arg callable returning the scalar
            energy, closed over the decomposition, layout and problem data.
    """

    def __init__(self, decomposition: TensorDecomposition, field_layout, loss):
        super().__init__()
        self.decomposition = decomposition
        self.field_layout = field_layout
        self.loss = loss
        decomposition.register_into(field_layout)

    def forward(self):
        self.decomposition.fill(self.field_layout)
        return self.loss()
```

Update `src/neurom/decompositions/__init__.py` to:

```python
from neurom.decompositions.base import TensorDecomposition
from neurom.decompositions.pgd import Axis, CPPGD
from neurom.decompositions.pgd_fem_model import PGDFEMModel
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (all unit tests including the two new `pgdfemmodel` tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd_fem_model.py src/neurom/decompositions/__init__.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): format-agnostic PGDFEMModel over TensorDecomposition"
```

---

### Task 4: Migrate the parametric-beam integration test to the layout flow

Rewrite the beam integration test so both cases solve through `PGDFEMModel` + `FieldLayout`, with the external energy reading via `separated_view`. Same physics, same analytical check.

**Files:**
- Modify: `tests/integration/test_1d_beam_deflection_PGD_test.py` (energy signature + both test bodies + imports)

**Interfaces:**
- Consumes: `PGDFEMModel(decomposition, field_layout, loss)` (Task 3); `CPPGD.separated_view`, `CPPGD.assemble`, `CPPGD.add_mode`, `CPPGD.freeze_mode`; `FieldLayout`; `jacobian_field`, `integrate`.
- Produces: nothing new (test-only).

- [ ] **Step 1: Rewrite the test (this is the failing state — it imports/uses APIs the file does not yet call)**

Replace the entire contents of `tests/integration/test_1d_beam_deflection_PGD_test.py` with:

```python
import pytest
import torch

from neurom.decompositions import Axis, CPPGD, PGDFEMModel
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import Dirichlet, NoConstraint
from neurom.differential import jacobian_field
from neurom.integrate import integrate
from neurom.field_layout import FieldLayout

torch.set_default_dtype(torch.float32)


def build_axis(name, coords, constraint, init_values):
    n = coords.shape[0]
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(nodes, elements)
    positions = Field(name=f"{name}_positions", topology=topology, values=coords)
    sf = LinearSegment()
    return Axis(
        name=name,
        nodes_positions=positions,
        sf=sf,
        mapping=IsoparametricMapping1D(sf),
        quad=TwoPoints1D(),
        constraint=constraint,
        init_values=init_values,
    )


def potential_energy(cppgd, field_layout, f_value):
    """External separable parametric energy, read from the filled FieldLayout."""
    sep = cppgd.separated_view(field_layout)
    space = sep["space"]
    para = sep["E"]
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


class Test1dBeamDeflectionPGD:
    relative_tolerance: float = 5e-2

    def test_parametric_beam_matches_analytical(self):
        x_min, x_max = 0.0, 10.0
        E_min, E_max = 100.0, 1000.0
        N_x, N_E = 40, 20
        f_value = 1000.0

        x_coords = torch.linspace(x_min, x_max, N_x).unsqueeze(-1)
        E_coords = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)

        space_axis = build_axis(
            "space",
            x_coords,
            Dirichlet(nodes=[0, N_x - 1], values_imposed=torch.zeros(2, 1)),
            init_values=torch.zeros(N_x, 1),   # space monom starts at 0
        )
        para_axis = build_axis(
            "E",
            E_coords,
            NoConstraint(),
            init_values=torch.ones(N_E, 1),    # E monom starts at 1 (gradient flows)
        )

        cppgd = CPPGD(axes=[space_axis, para_axis], n_modes_max=1, n_modes_ini=1)
        field_layout = FieldLayout()
        model = PGDFEMModel(
            cppgd, field_layout, lambda: potential_energy(cppgd, field_layout, f_value)
        )

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward(retain_graph=True)
            return loss

        for _ in range(30):
            optimizer.step(closure)

        # Compare assembled u(x, E) to the analytical parametric deflection.
        x_test = torch.linspace(x_min, x_max, 15)
        E_test = torch.linspace(E_min, E_max, 5)
        u = cppgd.assemble([x_test, E_test])          # (15, 5)

        xx = x_test.unsqueeze(-1)                      # (15, 1)
        EE = E_test.unsqueeze(0)                       # (1, 5)
        u_analytical = 0.5 * f_value * (xx - x_min) * (xx - x_max) / EE

        scale = float(u_analytical.abs().max())
        assert u.detach().numpy() == pytest.approx(
            u_analytical.numpy(), abs=self.relative_tolerance * scale
        )

    def test_greedy_enrichment_second_mode_stays_bounded(self):
        """Rank-1 analytical field must still hold after a greedy mode-2 enrichment.

        Adding a second mode should not perturb the already-converged rank-1
        solution: the enrichment should drive the extra mode towards ~0, so the
        rank-2 assembled field must still match the analytical rank-1 solution
        within tolerance.
        """
        x_min, x_max = 0.0, 10.0
        E_min, E_max = 100.0, 1000.0
        N_x, N_E = 40, 20
        f_value = 1000.0

        x_coords = torch.linspace(x_min, x_max, N_x).unsqueeze(-1)
        E_coords = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)

        space_axis = build_axis(
            "space",
            x_coords,
            Dirichlet(nodes=[0, N_x - 1], values_imposed=torch.zeros(2, 1)),
            init_values=torch.zeros(N_x, 1),
        )
        para_axis = build_axis(
            "E",
            E_coords,
            NoConstraint(),
            init_values=torch.ones(N_E, 1),
        )

        cppgd = CPPGD(axes=[space_axis, para_axis], n_modes_max=2, n_modes_ini=1)
        field_layout = FieldLayout()
        model = PGDFEMModel(
            cppgd, field_layout, lambda: potential_energy(cppgd, field_layout, f_value)
        )

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward(retain_graph=True)
            return loss

        for _ in range(30):
            optimizer.step(closure)

        # Greedy-enrich with a second mode: freeze mode 0, activate+zero mode 1.
        cppgd.freeze_mode(0)
        cppgd.add_mode()

        optimizer2 = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=200, line_search_fn="strong_wolfe",
        )

        def closure2():
            optimizer2.zero_grad()
            loss = model()
            loss.backward(retain_graph=True)
            return loss

        for _ in range(150):
            optimizer2.step(closure2)

        # Compare assembled u(x, E) (now rank-2) to the analytical rank-1 field.
        x_test = torch.linspace(x_min, x_max, 15)
        E_test = torch.linspace(E_min, E_max, 5)
        u = cppgd.assemble([x_test, E_test])          # (15, 5)

        xx = x_test.unsqueeze(-1)                      # (15, 1)
        EE = E_test.unsqueeze(0)                       # (1, 5)
        u_analytical = 0.5 * f_value * (xx - x_min) * (xx - x_max) / EE

        scale = float(u_analytical.abs().max())
        assert u.detach().numpy() == pytest.approx(
            u_analytical.numpy(), abs=self.relative_tolerance * scale
        )
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/integration/test_1d_beam_deflection_PGD_test.py -v`
Expected: PASS — both `test_parametric_beam_matches_analytical` and `test_greedy_enrichment_second_mode_stays_bounded` (assembled `u(x,E)` matches `0.5·f·(x−x_min)(x−x_max)/E` within 5%).

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest`
Expected: PASS — full suite green (same count as before this change; two PGD integration tests + all unit tests).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_1d_beam_deflection_PGD_test.py
git commit -m "test(integration): parametric beam PGD via PGDFEMModel + FieldLayout"
```

---

### Task 5: Docs — CHANGELOG + handoff notes

Record the change per CLAUDE.md and refresh the handoff notes' API section.

**Files:**
- Modify: `CHANGELOG.md` (new entry on top)
- Modify: `docs/CP_PGD_IMPLEMENTATION_NOTES.md` (API section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the CHANGELOG entry**

In `CHANGELOG.md`, insert this block immediately below the `Newest entries on top ...` header paragraph and above the existing `## 2026-07-09 — CP-PGD mode management tweaks` entry:

```markdown
## 2026-07-10 — CP-PGD on the FieldLayout / FEMModel abstraction

Branch `pgd_addition_solal`.

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
```

- [ ] **Step 2: Refresh the handoff notes**

In `docs/CP_PGD_IMPLEMENTATION_NOTES.md`, update the `CPPGD` bullet list in the "Files → Added — production code" section: remove the `interpolate_separated()` bullet and add bullets describing `register_into(field_layout)`, `fill(field_layout)`, `separated_view(field_layout)`, the new `TensorDecomposition` base class, and the `PGDFEMModel`. Keep the `assemble` and greedy-enrichment bullets. Add one line to the "How to run" section:

```markdown
uv run pytest tests/unit/decompositions/test_pgd.py -v          # module unit tests
```

(already present) and confirm the integration command still reads
`tests/integration/test_1d_beam_deflection_PGD_test.py`.

- [ ] **Step 3: Verify the suite is still green (docs shouldn't change it, but confirm)**

Run: `uv run pytest`
Expected: PASS — unchanged from Task 4.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md docs/CP_PGD_IMPLEMENTATION_NOTES.md
git commit -m "docs(decompositions): record FieldLayout/PGDFEMModel integration"
```

---

## Self-Review

**1. Spec coverage:**
- ABC seam (`TensorDecomposition`, `register_into`/`fill`) → Task 1. ✅
- `CPPGD.register_into` (all modes), `fill` (active), param-dedup behaviour → Task 1 (dedup exercised implicitly by `model.parameters()` in Tasks 3–4). ✅
- `separated_view` + drop `interpolate_separated` (canonical layout flow) → Task 2. ✅
- `PGDFEMModel` in `decompositions/`, exported, depends only on ABC → Task 3. ✅
- Format-agnostic guarantee via fake decomposition → Task 3. ✅
- External energy refactored to `separated_view`; beam solves through `PGDFEMModel`+`FieldLayout`; both rank-1 and greedy tests → Task 4. ✅
- No `FieldLayout` changes → enforced by Global Constraints; every method used (`add`/`update`/`__getitem__`) already exists. ✅
- Docs: CHANGELOG + notes → Task 5. ✅

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; every run step shows the command and expected result. ✅

**3. Type consistency:** `register_into(field_layout)`, `fill(field_layout)`, `separated_view(field_layout) -> dict[str, list[QuadratureAssemblyResult]]`, `PGDFEMModel(decomposition, field_layout, loss)` are named identically across the ABC (Task 1), `CPPGD` (Tasks 1–2), the model (Task 3), and the tests (Tasks 1–4). Monom key = `TrainableField.name` = `f"{axis.name}_mode{m}"` is used consistently. ✅
