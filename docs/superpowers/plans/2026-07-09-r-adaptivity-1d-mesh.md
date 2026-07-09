# 1D r-adaptivity (trainable node positions) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let 1D mesh node positions be optimized during training (r-adaptivity), so interior nodes migrate toward high-curvature regions while endpoints stay fixed, validated on a 1D Poisson problem.

**Architecture:** Add one new field-like provider, `TrainablePositions1D`, that parametrizes node coordinates through a monotone reparametrization (`softplus → cumsum → normalize`) so ordering and endpoints hold by construction. Plug it into the existing `Mesh`/`Topology`/`QuadratureContext` pipeline via a `Mesh` factory, and wire a per-forward geometry refresh in `FEMModel` so autograd reaches the position parameters.

**Tech Stack:** Python, PyTorch (`nn.Module`, autograd), pytest, `uv` for running.

## Global Constraints

- Run tests with `uv run pytest` (module import mode `importlib`; `testpaths=["tests"]`).
- Default dtype in tests is `torch.float32` (set explicitly at the top of each new test file, matching existing tests).
- The reparametrization is **1D only**; do not touch or generalize the existing `Topology` + free-position path — it is the intended route for future 2D r-adaptivity.
- Follow existing package layout: new field lives in `src/neurom/fields/` and is exported from `neurom.fields`.
- After the feature is complete, append a `CHANGELOG.md` entry (newest on top) — per repo `CLAUDE.md`.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Reparametrization math (verbatim): `δ = softplus(c)` (shape `n-1`), `x̃ = cat([0], cumsum(δ))` (shape `n`, so `x̃[0]=0`), `x = a + (b-a)·x̃/x̃[-1]` (shape `n`). Inverse for init: `increments = diff(initial)/(b-a)`, `c = inv_softplus(increments)`, with `inv_softplus(y) = log(-expm1(-y)) + y`.

---

## File Structure

- `src/neurom/fields/trainable_positions_1d.py` (new) — `inv_softplus` helper + `TrainablePositions1D(FieldBase)`. Sole responsibility: the 1D coordinate reparametrization and the `FieldBase` interface (`full_values`, `at_elements`, `dim`).
- `src/neurom/fields/__init__.py` (modify) — export `TrainablePositions1D`.
- `src/neurom/meshes/mesh.py` (modify) — add `Mesh.with_trainable_positions_1d` classmethod and `has_trainable_positions` property. No change to existing constructor/behaviour.
- `src/neurom/fem_model.py` (modify) — refresh geometry in `forward()` when the mesh has trainable positions.
- `tests/unit/fields/test_trainable_positions_1d.py` (new) — unit tests for the provider.
- `tests/unit/meshes/test_mesh_trainable_positions.py` (new) — unit tests for factory + property.
- `tests/integration/test_trainable_mesh_gradients.py` (new) — wiring test: gradients reach node positions through `FEMModel.forward()`.
- `tests/integration/test_1d_poisson_r_adaptivity.py` (new) — the r-adaptivity acceptance test.
- `CHANGELOG.md` (modify) — feature entry.

---

## Task 1: `TrainablePositions1D` provider

**Files:**
- Create: `src/neurom/fields/trainable_positions_1d.py`
- Modify: `src/neurom/fields/__init__.py`
- Test: `tests/unit/fields/test_trainable_positions_1d.py`

**Interfaces:**
- Consumes: `neurom.fields.field_base.FieldBase` (abstract `full_values`, `at_elements`); `neurom.meshes.topology.Topology` (`.n_nodes`, `.connectivity`).
- Produces:
  - `inv_softplus(y: torch.Tensor) -> torch.Tensor`
  - `TrainablePositions1D(name: str, topology: Topology, initial_positions: torch.Tensor)`; `initial_positions` shape `(n,1)` or `(n,)`, strictly increasing. Attributes: `coordinates: nn.Parameter` shape `(n-1,)`, buffers `a`, `b` (0-dim). Methods: `full_values() -> (n,1)`, `at_elements() -> (n_elements, 2, 1)`, property `dim -> 1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/fields/test_trainable_positions_1d.py`:

```python
import torch
import torch.nn.functional as F

from neurom.meshes import Topology
from neurom.fields import TrainablePositions1D
from neurom.fields.trainable_positions_1d import inv_softplus

torch.set_default_dtype(torch.float32)


def make_topology(n):
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    return Topology(nodes, elements)


def test_inv_softplus_is_inverse_of_softplus():
    y = torch.tensor([0.01, 0.1, 0.5, 1.0, 5.0])
    x = inv_softplus(y)
    assert torch.allclose(F.softplus(x), y, atol=1e-6)


def test_full_values_reproduces_uniform_mesh():
    n = 11
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert torch.allclose(pos.full_values(), x, atol=1e-5)


def test_full_values_reproduces_nonuniform_mesh():
    n = 6
    x = torch.tensor([0.0, 0.05, 0.2, 0.5, 0.85, 1.0]).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert torch.allclose(pos.full_values(), x, atol=1e-5)


def test_coordinates_shape_is_n_minus_one():
    n = 8
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert pos.coordinates.shape == (n - 1,)
    assert pos.coordinates.requires_grad


def test_endpoints_fixed_and_strictly_increasing_for_arbitrary_params():
    n = 8
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    with torch.no_grad():
        pos.coordinates.add_(torch.randn_like(pos.coordinates))
    fv = pos.full_values().reshape(-1)
    assert fv[0].item() == 0.0
    assert torch.isclose(fv[-1], torch.tensor(1.0), atol=1e-6)
    assert bool((fv[1:] > fv[:-1]).all())


def test_gradients_flow_to_coordinates():
    n = 7
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    loss = (pos.full_values() ** 2).sum()
    loss.backward()
    assert pos.coordinates.grad is not None
    assert torch.isfinite(pos.coordinates.grad).all()


def test_at_elements_matches_gather_and_dim():
    n = 5
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    pos = TrainablePositions1D(name="p", topology=top, initial_positions=x)
    ae = pos.at_elements()
    assert ae.shape == (n - 1, 2, 1)
    assert torch.allclose(ae, pos.full_values()[top.connectivity])
    assert pos.dim == 1


def test_rejects_non_increasing_initial_positions():
    n = 4
    x = torch.tensor([0.0, 0.5, 0.4, 1.0]).reshape(-1, 1)
    try:
        TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-increasing initial positions")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/fields/test_trainable_positions_1d.py -q`
Expected: FAIL — `ImportError: cannot import name 'TrainablePositions1D'`.

- [ ] **Step 3: Implement the provider**

Create `src/neurom/fields/trainable_positions_1d.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurom.fields.field_base import FieldBase
from neurom.meshes.topology import Topology


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of ``softplus``.

    ``softplus(x) = log(1 + exp(x))``; this returns ``x`` such that
    ``softplus(x) = y`` for ``y > 0``. Equal to ``log(exp(y) - 1)`` but stable
    for large ``y``.
    """
    return torch.log(-torch.expm1(-y)) + y


class TrainablePositions1D(FieldBase):
    """Trainable 1D node positions via a monotone reparametrization.

    **1D ONLY.** Node coordinates are recovered from free real parameters
    ``c`` (one per interval) as::

        delta   = softplus(c)                  # (n-1,)  strictly positive widths
        x_tilde = cat([0], cumsum(delta))      # (n,)    strictly increasing, x_tilde[0] = 0
        x       = a + (b - a) * x_tilde / x_tilde[-1]   # (n,)  x[0] = a, x[-1] = b

    ``softplus`` keeps every interval width positive so nodes never cross; the
    affine normalization pins the endpoints to ``[a, b]``. Both properties hold
    for *any* value of ``c`` — they are structural, not enforced by the
    optimizer. This relies on a total order on a line and does not generalize to
    2D/3D; the general ``Topology`` + free-position path stays the route for
    higher dimensions.

    Args:
        name (str): Field name.
        topology (Topology): Topology whose ``n_nodes`` must equal the number of
            ``initial_positions``.
        initial_positions (torch.Tensor): Shape ``(n, 1)`` or ``(n,)``, strictly
            increasing. ``a`` and ``b`` are taken from its first/last entries and
            the raw parameters are initialized so ``full_values()`` reproduces it.
    """

    def __init__(self, name: str, topology: Topology, initial_positions: torch.Tensor):
        super().__init__(name=name, topology=topology)

        pos = initial_positions.reshape(-1).to(torch.get_default_dtype())
        n = pos.shape[0]
        if n != topology.n_nodes:
            raise ValueError(
                f"initial_positions has {n} entries but topology has "
                f"{topology.n_nodes} nodes."
            )
        if not bool((pos[1:] > pos[:-1]).all()):
            raise ValueError("initial_positions must be strictly increasing.")

        self.register_buffer("a", pos[0].detach().clone())
        self.register_buffer("b", pos[-1].detach().clone())

        # Normalize increments (forward map re-normalizes, so scale is free) and
        # invert softplus so full_values() reproduces the input mesh.
        differences = pos[1:] - pos[:-1]              # (n-1,)
        increments = differences / (self.b - self.a)  # (n-1,), sum to 1
        self.coordinates = nn.Parameter(inv_softplus(increments))

    @property
    def dim(self) -> int:
        return 1

    def full_values(self) -> torch.Tensor:
        """Reparametrized coordinates, shape ``(n, 1)`` (global view)."""
        delta = F.softplus(self.coordinates)                 # (n-1,), > 0
        x_tilde = torch.cumsum(delta, dim=0)                 # (n-1,)
        zero = torch.zeros(1, dtype=x_tilde.dtype, device=x_tilde.device)
        x_tilde = torch.cat([zero, x_tilde])                 # (n,), x_tilde[0] = 0
        x = self.a + (self.b - self.a) * (x_tilde / x_tilde[-1])
        return x.reshape(-1, 1)

    def at_elements(self) -> torch.Tensor:
        """Element-local view: ``full_values()[topology.connectivity]``."""
        return self.full_values()[self.topology.connectivity]
```

Modify `src/neurom/fields/__init__.py` to add the export (append after the existing lines):

```python
from neurom.fields.field_base import FieldBase
from neurom.fields.field import Field
from neurom.fields.trainable_field import TrainableField
from neurom.fields.trainable_positions_1d import TrainablePositions1D
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/fields/test_trainable_positions_1d.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/fields/trainable_positions_1d.py src/neurom/fields/__init__.py tests/unit/fields/test_trainable_positions_1d.py
git commit -m "feat(fields): add TrainablePositions1D reparametrized node positions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `Mesh` factory + `has_trainable_positions`

**Files:**
- Modify: `src/neurom/meshes/mesh.py`
- Test: `tests/unit/meshes/test_mesh_trainable_positions.py`

**Interfaces:**
- Consumes: `TrainablePositions1D` (Task 1); existing `Mesh(topology, nodes_positions)`.
- Produces:
  - `Mesh.with_trainable_positions_1d(topology, initial_positions, name="positions") -> Mesh` (classmethod).
  - `Mesh.has_trainable_positions -> bool` (property): `any(p.requires_grad for p in self.nodes_positions.parameters())`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/meshes/test_mesh_trainable_positions.py`:

```python
import torch

from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainablePositions1D

torch.set_default_dtype(torch.float32)


def make_topology(n):
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    return Topology(nodes, elements)


def test_factory_builds_trainable_mesh():
    n = 9
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    mesh = Mesh.with_trainable_positions_1d(top, x)
    assert isinstance(mesh.nodes_positions, TrainablePositions1D)
    assert mesh.topology is top
    assert mesh.nodes_positions.topology is top
    assert mesh.has_trainable_positions
    assert torch.allclose(mesh.nodes_positions.full_values(), x, atol=1e-5)


def test_fixed_mesh_reports_no_trainable_positions():
    n = 9
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    mesh = Mesh(topology=top, nodes_positions=Field(name="pos", topology=top, values=x))
    assert not mesh.has_trainable_positions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/meshes/test_mesh_trainable_positions.py -q`
Expected: FAIL — `AttributeError: type object 'Mesh' has no attribute 'with_trainable_positions_1d'`.

- [ ] **Step 3: Implement factory + property**

In `src/neurom/meshes/mesh.py`, add these two members inside `class Mesh` (e.g. immediately after `__init__`, before `elements_at`). Use a **local import** in the classmethod to avoid a circular import (`neurom.meshes` ↔ `neurom.fields`):

```python
    @classmethod
    def with_trainable_positions_1d(cls, topology, initial_positions, name="positions"):
        """Build a 1D mesh whose interior node positions are trainable.

        The endpoints stay fixed and nodes stay ordered by construction (see
        ``TrainablePositions1D``). The plain ``Mesh(topology, nodes_positions)``
        constructor is unchanged and used for fixed meshes.

        Args:
            topology (Topology): The mesh topology.
            initial_positions (torch.Tensor): Strictly increasing node
                coordinates, shape ``(n, 1)`` or ``(n,)``.
            name (str): Name of the positions field.

        Returns:
            Mesh: A mesh with a ``TrainablePositions1D`` positions provider.
        """
        from neurom.fields.trainable_positions_1d import TrainablePositions1D

        positions = TrainablePositions1D(
            name=name, topology=topology, initial_positions=initial_positions
        )
        return cls(topology=topology, nodes_positions=positions)

    @property
    def has_trainable_positions(self) -> bool:
        """Whether the node positions carry trainable parameters.

        Provider-agnostic: ``True`` for any positions field exposing a parameter
        with ``requires_grad`` (e.g. ``TrainablePositions1D``), ``False`` for a
        plain ``Field``.
        """
        return any(p.requires_grad for p in self.nodes_positions.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/meshes/test_mesh_trainable_positions.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neurom/meshes/mesh.py tests/unit/meshes/test_mesh_trainable_positions.py
git commit -m "feat(meshes): Mesh factory and query for trainable 1D node positions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `FEMModel.forward()` geometry refresh wiring

**Files:**
- Modify: `src/neurom/fem_model.py`
- Test: `tests/integration/test_trainable_mesh_gradients.py`

**Interfaces:**
- Consumes: `Mesh.has_trainable_positions` (Task 2); existing `IntegrationDomain.update_contexts()` and `QuadratureContext.update()`; existing physics/assembly stack.
- Produces: `FEMModel.forward()` recomputes geometry from current node positions when `self.mesh.has_trainable_positions`, so `loss.backward()` populates `mesh.nodes_positions.coordinates.grad`.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_trainable_mesh_gradients.py`:

```python
import torch

from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.constraints import Dirichlet
from neurom.field_layout import FieldLayout
from neurom.interpolation import QuadratureContext, QuadratureAssembly, IntegrationDomain
from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


def test_forward_refresh_lets_gradients_reach_node_positions():
    n = 5
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    top = Topology(nodes, elements)

    sf = LinearSegment()
    quad = TwoPoints1D()
    mapping = IsoparametricMapping1D(sf)

    layout = FieldLayout()
    u = layout.add(
        TrainableField(
            name="u",
            topology=top,
            init_values=0.5 * torch.ones(n, 1),
            constraint=Dirichlet(nodes=[0, n - 1], values_imposed=torch.zeros(2, 1)),
        )
    )
    f = layout.add(Field(name="load", topology=top, values=torch.ones(n, 1)))

    mesh = Mesh.with_trainable_positions_1d(top, x)

    physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)
    loss = PhysicsLoss(physics=physics, field_layout=layout)

    ctx = QuadratureContext(mesh, quad, mapping)
    domain = IntegrationDomain(
        [QuadratureAssembly(ctx, sf, u), QuadratureAssembly(ctx, sf, f)]
    )
    model = FEMModel(
        mesh=mesh, field_layout=layout, integration_domain=domain, loss=loss
    )

    assert model.mesh.has_trainable_positions

    out = model()
    out.backward()

    coords = mesh.nodes_positions.coordinates
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_trainable_mesh_gradients.py -q`
Expected: FAIL — `coords.grad is None` (geometry cached at construction; `forward()` never refreshes it, so `coordinates` is not in the loss graph).

- [ ] **Step 3: Implement the refresh wiring**

In `src/neurom/fem_model.py`, replace the `forward` method body so it refreshes contexts when the mesh is trainable:

```python
    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        if self.mesh.has_trainable_positions:
            self.integration_domain.update_contexts()

        self.integration_domain.interpolate_all(self.field_layout)

        return self.loss()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_trainable_mesh_gradients.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Verify fixed-mesh path is unchanged**

Run: `uv run pytest tests/integration/test_1d_beam_deflection.py -q`
Expected: PASS (unchanged — `has_trainable_positions` is `False` for the beam's `Field` positions, so the refresh branch is skipped).

- [ ] **Step 6: Commit**

```bash
git add src/neurom/fem_model.py tests/integration/test_trainable_mesh_gradients.py
git commit -m "feat(fem): refresh geometry in forward when node positions are trainable

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 1D Poisson r-adaptivity acceptance test

**Files:**
- Test: `tests/integration/test_1d_poisson_r_adaptivity.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: everything from Tasks 1–3 plus the existing high-level stack (`ElasticEnergy`, `LoadPotential`, `PhysicsLoss`, `IntegrationDomain`, `FEMModel`, `PointWiseInterpolator`).
- Produces: no new production API — an acceptance test asserting (1) lower L2 error vs. a fixed mesh, (2) interior nodes migrate toward the feature, (3) endpoints/ordering invariants hold.

**Notes for the implementer:**
- Manufactured solution `u*(x) = exp(-((x-0.5)/w)²)` (≈0 at both ends), source `f = -u*''` (derived below). Load is a `Field`; because nodes move, it is **re-sampled at the current node positions each forward** in the adaptive run (a fixed nodal load would ride its node index and drift from `u*`).
- `f`-values are the only quantity treated as data (detached, re-sampled under `no_grad`); the measure `dx` and `u` still carry position gradients, which is what drives adaptation.
- Primary robust signal is node migration; L2 improvement should follow. `W`, `n`, `n_epochs`, `max_iter` are **tunable** — if the strict L2 inequality is borderline, raise `n_epochs`/`max_iter` first (the manufactured feature is sharp, so clustering must reduce error).

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_1d_poisson_r_adaptivity.py`:

```python
import torch

from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.constraints import Dirichlet
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    PointWiseInterpolator,
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)

X0 = 0.5
W = 0.06


def u_star(x):
    return torch.exp(-((x - X0) / W) ** 2)


def f_analytic(x):
    # f = -u*'' for u* = exp(-((x-X0)/W)^2)
    g = torch.exp(-((x - X0) / W) ** 2)
    return g * (2.0 / W ** 2 - 4.0 * (x - X0) ** 2 / W ** 4)


def build(n, trainable):
    x_array = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    top = Topology(nodes, elements)

    sf = LinearSegment()
    quad = TwoPoints1D()
    mapping = IsoparametricMapping1D(sf)

    layout = FieldLayout()
    u = layout.add(
        TrainableField(
            name="u",
            topology=top,
            init_values=torch.zeros(n, 1),
            constraint=Dirichlet(nodes=[0, n - 1], values_imposed=torch.zeros(2, 1)),
        )
    )

    if trainable:
        mesh = Mesh.with_trainable_positions_1d(top, x_array)
    else:
        mesh = Mesh(
            topology=top,
            nodes_positions=Field(name="pos", topology=top, values=x_array),
        )

    f = layout.add(Field(name="load", topology=top, values=f_analytic(x_array)))

    physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)
    loss = PhysicsLoss(physics=physics, field_layout=layout)

    ctx = QuadratureContext(mesh, quad, mapping)
    domain = IntegrationDomain(
        [QuadratureAssembly(ctx, sf, u), QuadratureAssembly(ctx, sf, f)]
    )
    model = FEMModel(
        mesh=mesh, field_layout=layout, integration_domain=domain, loss=loss
    )
    return model, mesh, u, f, sf, mapping


def train(model, mesh, f, trainable, n_epochs=8, max_iter=100):
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        if trainable:
            with torch.no_grad():
                xq = mesh.nodes_positions.full_values().reshape(-1, 1)
                f.values.copy_(f_analytic(xq))
        loss = model()
        loss.backward(retain_graph=True)
        return loss

    for _ in range(n_epochs):
        model()
        optimizer.step(closure)


def l2_error(mesh, u, sf, mapping, n_dense=400):
    x_dense = torch.linspace(0.0, 1.0, n_dense)
    pwi = PointWiseInterpolator(mesh, sf, u, mapping)
    u_h = pwi.at_position(x_dense).reshape(-1)
    err = u_h - u_star(x_dense)
    return torch.sqrt(torch.mean(err ** 2)).item()


def test_poisson_r_adaptivity_improves_accuracy_and_moves_nodes():
    torch.manual_seed(0)
    n = 21

    # Fixed-mesh baseline: train displacement only.
    model_fix, mesh_fix, u_fix, f_fix, sf_fix, map_fix = build(n, trainable=False)
    train(model_fix, mesh_fix, f_fix, trainable=False)
    err_fixed = l2_error(mesh_fix, u_fix, sf_fix, map_fix)

    # Adaptive mesh: train displacement + node positions jointly.
    model_ad, mesh_ad, u_ad, f_ad, sf_ad, map_ad = build(n, trainable=True)
    x_init = mesh_ad.nodes_positions.full_values().reshape(-1).detach().clone()
    train(model_ad, mesh_ad, f_ad, trainable=True)
    err_adapt = l2_error(mesh_ad, u_ad, sf_ad, map_ad)
    x_final = mesh_ad.nodes_positions.full_values().reshape(-1).detach()

    # 1. Accuracy improves.
    assert err_adapt < err_fixed

    # 2. Interior nodes migrate toward the feature at x = 0.5.
    interior_init = x_init[1:-1]
    interior_final = x_final[1:-1]
    assert (interior_final - X0).abs().mean() < (interior_init - X0).abs().mean()

    # 3. Structural invariants (guaranteed by construction).
    assert torch.isclose(x_final[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(x_final[-1], torch.tensor(1.0), atol=1e-6)
    assert bool((x_final[1:] > x_final[:-1]).all())
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_1d_poisson_r_adaptivity.py -q`
Expected: PASS (1 test). If the `err_adapt < err_fixed` assertion is borderline, increase `n_epochs`/`max_iter` in `train(...)` (the feature is sharp enough that clustering must reduce error) before touching anything else. All other assertions are structural and must hold regardless.

- [ ] **Step 3: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: PASS — previously-passing tests plus the new ones (fixed-mesh paths untouched).

- [ ] **Step 4: Update CHANGELOG**

Prepend a new entry to `CHANGELOG.md` (newest on top), just under the title block:

```markdown
## 2026-07-09 — 1D r-adaptivity (trainable node positions)

Branch `develop_solal`. Design/plan:
`docs/superpowers/specs/2026-07-09-r-adaptivity-1d-mesh-design.md`,
`docs/superpowers/plans/2026-07-09-r-adaptivity-1d-mesh.md`.

- Added `TrainablePositions1D` (`src/neurom/fields/`): 1D node positions
  reparametrized as `softplus → cumsum → normalize`, so nodes stay ordered and
  endpoints stay fixed by construction (r-adaptivity, 1D only). The general
  `Topology` + free-position path is untouched for future 2D.
- `Mesh.with_trainable_positions_1d(...)` factory and `Mesh.has_trainable_positions`
  query.
- `FEMModel.forward()` refreshes quadrature geometry each forward when node
  positions are trainable, so autograd reaches the position parameters.
- Integration test: 1D Poisson with a sharp Gaussian feature; the adaptive mesh
  lowers the L2 error vs. a fixed mesh and clusters interior nodes toward the
  feature.
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_1d_poisson_r_adaptivity.py CHANGELOG.md
git commit -m "test(integration): 1D Poisson r-adaptivity acceptance test + changelog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Reparametrization (`softplus/cumsum/normalize`, inverse init, `inv_softplus`) → Task 1.
- `TrainablePositions1D` in `neurom/fields/`, `FieldBase` interface, `dim`, `at_elements` → Task 1.
- Round-trip covered by unit tests (uniform + non-uniform), no runtime assert in `__init__` → Task 1 (`test_full_values_reproduces_*`).
- 1D-only scoping / general path untouched → enforced by not modifying `Topology`; stated in docstring + Global Constraints.
- `Mesh.with_trainable_positions_1d` + `has_trainable_positions` (factory vs. query) → Task 2.
- `FEMModel.forward()` refresh gated on trainable positions; fixed path unchanged → Task 3 (with beam regression check).
- Poisson sharp-feature test: fixed vs. adaptive, load re-sampled at current nodes, L2 improvement + node migration + invariants → Task 4.
- CHANGELOG entry → Task 4.
- CP-PGD union-mesh caveat and node-crossing limitation are documented in the spec; no task needed (out of scope, structural guarantee removes the crossing risk here).

**Placeholder scan:** none — all steps carry concrete code and exact commands.

**Type consistency:** `TrainablePositions1D(name, topology, initial_positions)`, `full_values()→(n,1)`, `at_elements()→(n_elements,2,1)`, `coordinates` shape `(n-1,)`, `Mesh.with_trainable_positions_1d(topology, initial_positions, name=...)`, `Mesh.has_trainable_positions` — used identically across Tasks 1–4. `f.values.copy_(...)` matches `Field`'s `values` buffer.
