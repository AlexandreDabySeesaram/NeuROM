# CP-PGD Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure CP-PGD separated-representation model (`CPPGD`) plus a per-axis descriptor (`Axis`) to the modular `neurom` library, generalized to `l` axes, with greedy mode enrichment.

**Architecture:** `CPPGD` is an `nn.Module` holding a grid of monoms (`TrainableField` per mode × axis) on top of the existing FE building blocks (`Mesh`, `QuadratureContext`, `QuadratureAssembly`, `PointWiseInterpolator`). It exposes an assembled full-tensor view (`assemble`) and a per-monom separated view (`interpolate_separated`). The energy/loss and the greedy training loop live entirely outside the module (in the driver/test).

**Tech Stack:** Python ≥ 3.12, PyTorch, pytest, `uv` for running.

## Global Constraints

- Default dtype: `torch.float32` (call `torch.set_default_dtype(torch.float32)` at top of every test module, matching existing tests).
- Run everything through `uv` (e.g. `uv run pytest ...`); the bare `python`/`pytest` are not on PATH.
- Follow existing library patterns: `nn.Module` subclasses, `register_buffer` for non-trainable tensors, tests mirror `src/` layout under `tests/unit/`.
- The module MUST NOT compute any energy/loss and MUST NOT contain a training loop/helper. Both live in the driver/test.
- `interpolate_separated` exposes **each monom individually** (per-axis list indexed by mode), not a stacked tensor.
- The reference beam integration test goes in `tests/integration/test_1d_beam_deflection_PGD_test.py`. Do NOT touch `tests/integration/test_1d_beam_deflection_PGD.py` (reserved for the user).

---

## File Structure

- Create `src/neurom/decompositions/__init__.py` — exports `CPPGD`, `Axis`.
- Create `src/neurom/decompositions/pgd.py` — both `Axis` (dataclass) and `CPPGD` (`nn.Module`).
- Create `tests/unit/decompositions/test_pgd.py` — unit tests for `Axis` and `CPPGD`.
- Create `tests/integration/test_1d_beam_deflection_PGD_test.py` — reference parametric beam solve.

---

## Task 1: Package scaffold + `Axis` descriptor

**Files:**
- Create: `src/neurom/decompositions/__init__.py`
- Create: `src/neurom/decompositions/pgd.py`
- Test: `tests/unit/decompositions/test_pgd.py`

**Interfaces:**
- Produces:
  - `Axis` dataclass with fields `name: str`, `nodes_positions` (a `neurom.fields.Field`), `sf` (`ShapeFunction`), `mapping`, `quad` (`QuadratureRule`), `constraint` (`Constraint`), `init_values: torch.Tensor`; and a read-only property `topology` returning `nodes_positions.topology`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/decompositions/test_pgd.py`:

```python
import pytest
import torch

from neurom.decompositions import Axis
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import NoConstraint

torch.set_default_dtype(torch.float32)


def make_axis(name="space", n=5, lo=0.0, hi=10.0):
    coords = torch.linspace(lo, hi, n).unsqueeze(-1)
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
        constraint=NoConstraint(),
        init_values=torch.zeros(n, 1),
    )


def test_axis_exposes_topology_from_positions():
    axis = make_axis()
    assert axis.name == "space"
    # topology property must be the SAME object as the positions' topology
    assert axis.topology is axis.nodes_positions.topology
    assert axis.topology.n_nodes == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_axis_exposes_topology_from_positions -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurom.decompositions'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/neurom/decompositions/pgd.py`:

```python
from dataclasses import dataclass

import torch

from neurom.fields.field import Field
from neurom.shape_functions.shape_function import ShapeFunction
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.constraints.constraint import Constraint
from neurom.meshes.topology import Topology


@dataclass
class Axis:
    """Descriptor of one factor (coordinate direction) of a CP-PGD decomposition.

    Groups everything needed to build and interpolate the monoms living on this
    axis. The axis mesh topology is derived from ``nodes_positions`` so that the
    ``Mesh`` identity check and the monoms' ``TrainableField`` share the exact
    same ``Topology`` object.

    Attributes:
        name (str): Axis name, used as key in the separated interpolation output.
        nodes_positions (Field): Coordinates of the axis mesh nodes.
        sf (ShapeFunction): Shape function used for interpolation on this axis.
        mapping: Reference/physical mapping (e.g. IsoparametricMapping1D).
        quad (QuadratureRule): Quadrature rule for integration on this axis.
        constraint (Constraint): Constraint (boundary conditions) on this axis.
        init_values (torch.Tensor): Initial nodal values for each new monom,
            shape (n_nodes, dim).
    """

    name: str
    nodes_positions: Field
    sf: ShapeFunction
    mapping: object
    quad: QuadratureRule
    constraint: Constraint
    init_values: torch.Tensor

    @property
    def topology(self) -> Topology:
        return self.nodes_positions.topology
```

Create `src/neurom/decompositions/__init__.py`:

```python
from neurom.decompositions.pgd import Axis, CPPGD
```

Note: `CPPGD` is added in Task 2. To keep this task importable on its own, temporarily export only `Axis`:

```python
from neurom.decompositions.pgd import Axis
```

(Task 2 will extend this line to also export `CPPGD`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_axis_exposes_topology_from_positions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/__init__.py src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): add Axis descriptor for CP-PGD"
```

---

## Task 2: `CPPGD` construction & initial freeze state

**Files:**
- Modify: `src/neurom/decompositions/pgd.py`
- Modify: `src/neurom/decompositions/__init__.py`
- Test: `tests/unit/decompositions/test_pgd.py`

**Interfaces:**
- Consumes: `Axis` (Task 1).
- Produces:
  - `CPPGD(axes: list[Axis], n_modes_max: int, n_modes_ini: int = 1)` — `nn.Module`.
  - Attributes: `self.axes` (list), `self.n_modes_max` (int), `self.n_modes_truncated` (int-valued buffer, currently active modes), `self.monoms` (`nn.ModuleList` over modes of `nn.ModuleList` over axes of `TrainableField`), `self._meshes` (`nn.ModuleList[Mesh]`), `self._contexts` (`nn.ModuleList[QuadratureContext]`).
  - Only monoms of modes `0 .. n_modes_truncated-1` are trainable at construction; the rest are frozen.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/decompositions/test_pgd.py`:

```python
from neurom.decompositions import CPPGD


def make_two_axes():
    space = make_axis(name="space", n=5, lo=0.0, hi=10.0)
    para = make_axis(name="E", n=4, lo=100.0, hi=1000.0)
    return [space, para]


def test_cppgd_construction_structure_and_freeze():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=3, n_modes_ini=1)

    # 3 modes, each with 2 monoms (one per axis)
    assert len(model.monoms) == 3
    assert all(len(mode) == 2 for mode in model.monoms)

    # Only mode 0 active
    assert int(model.n_modes_truncated) == 1

    # Mode 0 monoms trainable, modes 1 and 2 frozen
    assert all(f.values_reduced.requires_grad for f in model.monoms[0])
    assert all(not f.values_reduced.requires_grad for f in model.monoms[1])
    assert all(not f.values_reduced.requires_grad for f in model.monoms[2])

    # Active parameters == the 2 monoms of mode 0
    active = [p for p in model.parameters() if p.requires_grad]
    assert len(active) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_cppgd_construction_structure_and_freeze -v`
Expected: FAIL with `ImportError: cannot import name 'CPPGD'`.

- [ ] **Step 3: Write minimal implementation**

In `src/neurom/decompositions/pgd.py`, add imports at the top:

```python
import torch.nn as nn

from neurom.fields.trainable_field import TrainableField
from neurom.meshes.mesh import Mesh
from neurom.interpolation.quadrature_context import QuadratureContext
```

Then add the class:

```python
class CPPGD(nn.Module):
    """Canonical-polyadic PGD separated-representation model.

    Represents ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` over ``l`` axes. Holds the
    monoms ``w_m^k`` as ``TrainableField`` on each axis, manages greedy mode
    enrichment, and exposes an assembled full-tensor view and a per-monom
    separated view. It computes no energy and owns no training loop.

    Args:
        axes (list[Axis]): The ordered axes of the decomposition.
        n_modes_max (int): Maximum number of modes.
        n_modes_ini (int): Number of initially active (trainable) modes.
    """

    def __init__(self, axes, n_modes_max, n_modes_ini=1):
        super().__init__()
        self.axes = list(axes)
        self.n_modes_max = n_modes_max
        self.register_buffer(
            "n_modes_truncated", torch.tensor(min(n_modes_ini, n_modes_max))
        )

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
                            name=f"{a.name}_mode{m}",
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

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(int(self.n_modes_truncated)):
            self.unfreeze_mode(m)

    def freeze_all(self):
        """Freeze the monoms of every mode."""
        for m in range(self.n_modes_max):
            self.freeze_mode(m)

    def freeze_mode(self, m):
        """Freeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(False)

    def unfreeze_mode(self, m):
        """Unfreeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(True)
```

Update `src/neurom/decompositions/__init__.py`:

```python
from neurom.decompositions.pgd import Axis, CPPGD
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd.py src/neurom/decompositions/__init__.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): add CPPGD construction and freeze state"
```

---

## Task 3: `interpolate_separated` (per-monom separated view)

**Files:**
- Modify: `src/neurom/decompositions/pgd.py`
- Test: `tests/unit/decompositions/test_pgd.py`

**Interfaces:**
- Consumes: `CPPGD` (Task 2), `QuadratureAssembly`, `QuadratureAssemblyResult`.
- Produces:
  - `CPPGD.interpolate_separated() -> dict[str, list[QuadratureAssemblyResult]]`. For each axis name, a list of length `n_modes_truncated`; entry `m` is the interpolation of monom `w_m^axis` at that axis's quadrature points, with `u` of shape `(N_e, N_q, u_dim)` and shared `x`, `measure`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/decompositions/test_pgd.py`:

```python
from neurom.interpolation.quadrature_assembly import QuadratureAssembly


def test_interpolate_separated_keys_shapes_and_values():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=2)

    # Give mode-0 space monom known nodal values so we can predict the result.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.ones_like(model.monoms[0][0].values_reduced)
        )

    sep = model.interpolate_separated()

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_interpolate_separated_keys_shapes_and_values -v`
Expected: FAIL with `AttributeError: 'CPPGD' object has no attribute 'interpolate_separated'`.

- [ ] **Step 3: Write minimal implementation**

Add import at top of `pgd.py`:

```python
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
```

Add the method to `CPPGD`:

```python
    def interpolate_separated(self):
        """Interpolate each active monom at its axis's quadrature points.

        Returns:
            dict[str, list[QuadratureAssemblyResult]]: axis name -> list indexed
            by mode; entry ``m`` is the interpolation of the single monom
            ``w_m^axis``. Enables writing separable energies monom by monom.
        """
        result = {}
        for k, axis in enumerate(self.axes):
            ctx = self._contexts[k]
            per_mode = []
            for m in range(int(self.n_modes_truncated)):
                assembly = QuadratureAssembly(ctx, axis.sf, self.monoms[m][k])
                per_mode.append(assembly.interpolate())
            result[axis.name] = per_mode
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): add per-monom interpolate_separated to CPPGD"
```

---

## Task 4: `assemble` (full assembled tensor)

**Files:**
- Modify: `src/neurom/decompositions/pgd.py`
- Test: `tests/unit/decompositions/test_pgd.py`

**Interfaces:**
- Consumes: `CPPGD` (Task 2), `PointWiseInterpolator`.
- Produces:
  - `CPPGD.assemble(coords: list[torch.Tensor]) -> torch.Tensor`. `coords` has one 1-D tensor per axis (length `N_k`). Returns the full tensor of shape `(N_1, ..., N_l)` equal to `sum_m prod_k w_m^k(coords[k])`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/decompositions/test_pgd.py`:

```python
def test_assemble_matches_manual_outer_product():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)

    # NoConstraint on both axes -> nodal values are the full field. Set them.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)  # S at 5 space nodes
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)  # g at 4 E nodes
        )

    x = torch.tensor([2.5, 5.0])          # inside space domain [0, 10]
    E = torch.tensor([400.0, 700.0])      # inside E domain [100, 1000]
    u = model.assemble([x, E])

    assert u.shape == (2, 2)

    # Manual: interpolate each monom pointwise, then outer product (single mode).
    from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

    pwi_s = PointWiseInterpolator(
        model._meshes[0], axes[0].sf, model.monoms[0][0], axes[0].mapping
    )
    pwi_e = PointWiseInterpolator(
        model._meshes[1], axes[1].sf, model.monoms[0][1], axes[1].mapping
    )
    s = pwi_s.at_position(x).reshape(-1)
    g = pwi_e.at_position(E).reshape(-1)
    expected = torch.outer(s, g)
    assert torch.allclose(u, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_assemble_matches_manual_outer_product -v`
Expected: FAIL with `AttributeError: 'CPPGD' object has no attribute 'assemble'`.

- [ ] **Step 3: Write minimal implementation**

Add imports at top of `pgd.py`:

```python
import string

from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
```

Add the method to `CPPGD`:

```python
    def assemble(self, coords):
        """Assemble the full separated tensor at the given per-axis coordinates.

        Args:
            coords (list[torch.Tensor]): One 1-D tensor per axis (length N_k),
                the query coordinates on that axis.

        Returns:
            torch.Tensor: Full tensor of shape (N_1, ..., N_l) equal to
            ``sum_m prod_k w_m^k(coords[k])``. Detached (for post-processing).
        """
        n_modes = int(self.n_modes_truncated)
        per_axis = []  # per_axis[k]: (n_modes, N_k)
        for k, axis in enumerate(self.axes):
            mesh = self._meshes[k]
            cols = []
            for m in range(n_modes):
                pwi = PointWiseInterpolator(mesh, axis.sf, self.monoms[m][k], axis.mapping)
                cols.append(pwi.at_position(coords[k].reshape(-1)).reshape(-1))
            per_axis.append(torch.stack(cols, dim=0))

        n_axes = len(self.axes)
        axis_letters = string.ascii_lowercase[:n_axes]
        mode_letter = "Z"
        in_subs = ",".join(mode_letter + axis_letters[k] for k in range(n_axes))
        out_subs = axis_letters
        return torch.einsum(f"{in_subs}->{out_subs}", *per_axis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): add full-tensor assemble to CPPGD"
```

---

## Task 5: Greedy mode management (`add_mode`, `add_mode_to_optimizer`)

**Files:**
- Modify: `src/neurom/decompositions/pgd.py`
- Test: `tests/unit/decompositions/test_pgd.py`

**Interfaces:**
- Consumes: `CPPGD` (Task 2).
- Produces:
  - `CPPGD.add_mode()` — freeze the currently-active modes, increment `n_modes_truncated`, zero-out and unfreeze the newly activated mode's monoms. Raises `RuntimeError` if already at `n_modes_max`.
  - `CPPGD.add_mode_to_optimizer(optim: torch.optim.Optimizer)` — add the newly-activated mode's monom parameters to `optim` via `add_param_group`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/decompositions/test_pgd.py`:

```python
def test_add_mode_freezes_previous_and_activates_new():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=1)

    # Dirty the (frozen) mode-1 monoms so we can check zero-out.
    with torch.no_grad():
        for f in model.monoms[1]:
            f.values_reduced.add_(7.0)

    model.add_mode()

    assert int(model.n_modes_truncated) == 2
    # Previous mode frozen, new mode active
    assert all(not f.values_reduced.requires_grad for f in model.monoms[0])
    assert all(f.values_reduced.requires_grad for f in model.monoms[1])
    # New mode zeroed out
    assert all(torch.count_nonzero(f.values_reduced) == 0 for f in model.monoms[1])


def test_add_mode_raises_at_max():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=1, n_modes_ini=1)
    with pytest.raises(RuntimeError):
        model.add_mode()


def test_add_mode_to_optimizer_grows_param_groups():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=1)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    n_before = sum(len(g["params"]) for g in optim.param_groups)
    model.add_mode()
    model.add_mode_to_optimizer(optim)
    n_after = sum(len(g["params"]) for g in optim.param_groups)
    # 2 new monom parameters (one per axis) added
    assert n_after == n_before + 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py::test_add_mode_freezes_previous_and_activates_new -v`
Expected: FAIL with `AttributeError: 'CPPGD' object has no attribute 'add_mode'`.

- [ ] **Step 3: Write minimal implementation**

Add methods to `CPPGD`:

```python
    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Freezes the currently-active modes, activates the next mode (zeroed out
        and trainable). Raises RuntimeError if already at n_modes_max.
        """
        if int(self.n_modes_truncated) >= self.n_modes_max:
            raise RuntimeError(
                f"Cannot add mode: already at n_modes_max={self.n_modes_max}."
            )
        for m in range(int(self.n_modes_truncated)):
            self.freeze_mode(m)
        new = int(self.n_modes_truncated)
        self.n_modes_truncated += 1
        self._zero_out(new)
        self.unfreeze_mode(new)

    def _zero_out(self, m):
        """Zero the nodal values of every monom of mode ``m``."""
        with torch.no_grad():
            for field in self.monoms[m]:
                field.values_reduced.zero_()

    def add_mode_to_optimizer(self, optim):
        """Add the last-activated mode's monom parameters to ``optim``."""
        new = int(self.n_modes_truncated) - 1
        params = [f.values_reduced for f in self.monoms[new]]
        optim.add_param_group({"params": params})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/decompositions/test_pgd.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/decompositions/pgd.py tests/unit/decompositions/test_pgd.py
git commit -m "feat(decompositions): add greedy mode management to CPPGD"
```

---

## Task 6: Reference integration test — parametric beam solve

**Files:**
- Create: `tests/integration/test_1d_beam_deflection_PGD_test.py`

**Interfaces:**
- Consumes: `CPPGD`, `Axis` (Tasks 1–5), `jacobian_field`, `integrate`.
- Produces: an end-to-end test demonstrating the **external** parametric energy
  written monom-by-monom from `interpolate_separated()`, solved with LBFGS,
  matching the analytical parametric deflection.

**Physics reference:** With the existing library sign convention
(`ElasticEnergy` is `+½(u')²`, `LoadPotential` contributes `+f·u` after the
`ElasticEnergy - LoadPotential` combination), the parametric potential energy is

```
J = 0.5 * ∫_E E ∫_x (∂_x u)^2 dx dE  +  ∫_E ∫_x f u dx dE ,   u(x,E) = Σ_m S_m(x) g_m(E)
```

whose minimizer is `u(x,E) = 0.5 * f * (x - x_min) * (x - x_max) / E`
(rank-1: `S_0 = 0.5 f (x-x_min)(x-x_max)`, `g_0 = 1/E`).

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_1d_beam_deflection_PGD_test.py`:

```python
import pytest
import torch

from neurom.decompositions import Axis, CPPGD
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import Dirichlet, NoConstraint
from neurom.differential import jacobian_field
from neurom.integrate import integrate

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


def potential_energy(model, f_value):
    """External separable parametric energy, written monom by monom."""
    sep = model.interpolate_separated()
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
            init_values=torch.ones(N_E, 1),    # E monom starts at 1 (nonzero -> gradient flows)
        )

        model = CPPGD(axes=[space_axis, para_axis], n_modes_max=1, n_modes_ini=1)

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = potential_energy(model, f_value)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(30):
            optimizer.step(closure)

        # Compare assembled u(x, E) to the analytical parametric deflection.
        x_test = torch.linspace(x_min, x_max, 15)
        E_test = torch.linspace(E_min, E_max, 5)
        u = model.assemble([x_test, E_test])          # (15, 5)

        xx = x_test.unsqueeze(-1)                      # (15, 1)
        EE = E_test.unsqueeze(0)                       # (1, 5)
        u_analytical = 0.5 * f_value * (xx - x_min) * (xx - x_max) / EE

        scale = float(u_analytical.abs().max())
        assert u.detach().numpy() == pytest.approx(
            u_analytical.numpy(), abs=self.relative_tolerance * scale
        )
```

- [ ] **Step 2: Run test to verify it fails, then converges**

Run: `uv run pytest tests/integration/test_1d_beam_deflection_PGD_test.py -v`
Expected: initially may FAIL on the tolerance if the solve under-converges.
If so, increase the epoch count (the `for _ in range(30)` loop) and/or
`max_iter` until the assembled field matches within `5e-2`. This is an
iterative solver: tuning iteration counts (not the physics or tolerance) to
reach convergence is the intended TDD adjustment here. Do not loosen the
tolerance below what demonstrates a correct solve.

- [ ] **Step 3: Confirm the full suite is green**

Run: `uv run pytest tests/unit/decompositions tests/integration/test_1d_beam_deflection_PGD_test.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_1d_beam_deflection_PGD_test.py
git commit -m "test(integration): parametric beam PGD reference solve"
```

---

## Self-Review

**Spec coverage:**
- Pure separated-representation model generalized to `l` axes → Tasks 2, 4 (`assemble` uses a dynamic einsum over any number of axes).
- `Axis` descriptor (renamed from `PGDAxis`) → Task 1.
- Greedy enrichment (add mode, freeze previous, unfreeze/zero new) faithful to `NeuROM` → Task 5.
- Two views: assembled full tensor + per-monom separated → Tasks 3 (per-monom), 4 (assembled).
- Energy and training loop OUTSIDE the module → enforced by Global Constraints; demonstrated in Task 6.
- `interpolate_separated` exposes each monom individually → Task 3 returns `dict[str, list[QuadratureAssemblyResult]]`.
- Reference test in `test_1d_beam_deflection_PGD_test.py`, leaving `test_1d_beam_deflection_PGD.py` untouched → Task 6.
- Homogeneous BCs via each axis `Constraint`; no relevement mode → space axis uses `Dirichlet`, E axis `NoConstraint` (Task 6); no lifting logic added (deferred, per spec non-goals).

**Placeholder scan:** No TBD/TODO; all steps contain concrete code and commands. The only latitude is tuning iteration counts in Task 6 Step 2, which is inherent to testing an iterative solver and explicitly bounded (adjust iterations, not tolerance/physics).

**Type consistency:** `Axis` fields and `.topology` property consistent across Tasks 1/2/6. `CPPGD` method names — `freeze_all`, `freeze_mode`, `unfreeze_mode`, `interpolate_separated`, `assemble`, `add_mode`, `_zero_out`, `add_mode_to_optimizer`, and the `n_modes_truncated` buffer / `n_modes_max` attribute — used identically wherever referenced. `interpolate_separated` returns per-axis lists of `QuadratureAssemblyResult` (`.x`, `.u`, `.measure`), consumed exactly that way in Task 6.
