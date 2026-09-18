import pytest
import torch

from neurom.decompositions import CPPGD, TensorDecomposition
from neurom.field_layout import FieldLayout
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.interpolation import IntegrationDomain
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.math import integrate
from neurom.meshes import Connectivity, Mesh
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearBar

torch.set_default_dtype(torch.float32)


class _ConstantDecomposition(TensorDecomposition):
    """Minimal fake decomposition with NO CP structure: registers one fixed
    Field and exposes a QuadratureAssembly for it. It has no ``monoms``, no
    ``factors`` and no ``n_active_modes``, so any CP-specific attribute access
    that creeps into NeuROMModel raises AttributeError against it."""

    def __init__(self):
        super().__init__()
        n = 4
        coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
        nodes = torch.arange(0, n)
        elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
        conn = Connectivity(nodes, elements)
        positions = Field(name="dummy_pos", connectivity=conn, values=coords)
        self.field = Field(name="dummy", connectivity=conn, values=torch.ones(n, 1))
        sf = LinearBar()
        mesh = Mesh(connectivity=conn, nodes_positions=positions)
        ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(sf, mesh))
        self._assembly = QuadratureAssembly(ctx, sf, self.field)

    def register_into(self, field_layout):
        field_layout.add(self.field)

    def assemblies(self):
        return [self._assembly]

    def evaluate(self, coords):
        return torch.ones(coords[0].reshape(-1).shape[0], 1)

    def assemble(self, coords):
        return torch.ones(*[c.reshape(-1).shape[0] for c in coords])


def test_neurommodel_train_forward_returns_layout_and_optimizes(two_specs):
    cppgd = CPPGD(monom_specs=two_specs, n_modes_max=1, n_modes_ini=1)
    layout = FieldLayout()
    domain = IntegrationDomain(cppgd.assemblies())

    def energy(out):
        name = cppgd.directory()["space"][0]
        s = out[name]
        return integrate(
            s.u.values * s.measure.values
        )  # linear in S -> nonzero grad at 0 init

    model = NeuROMModel(layout, cppgd, domain, energy)
    out = model()  # training forward
    assert out is layout  # returns the filled layout

    before = cppgd.monoms[0][0].values_reduced.detach().clone()
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    optim.zero_grad()
    loss = model.loss(model())
    loss.backward()
    optim.step()
    after = cppgd.monoms[0][0].values_reduced.detach()
    assert not torch.allclose(before, after)


def test_neurommodel_add_mode_to_optimizer_grows_param_groups(two_specs):
    cppgd = CPPGD(monom_specs=two_specs, n_modes_max=2, n_modes_ini=1)
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, loss=lambda out: out)
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    n_before = sum(len(g["params"]) for g in optim.param_groups)
    cppgd.add_mode()
    model.add_mode_to_optimizer(optim)
    n_after = sum(len(g["params"]) for g in optim.param_groups)
    # 2 new monom parameters (one per factor) added
    assert n_after == n_before + 2


def test_neurommodel_eval_forward_matched_pointwise(two_specs):
    cppgd = CPPGD(monom_specs=two_specs, n_modes_max=1, n_modes_ini=1)
    with torch.no_grad():
        cppgd.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)
        )
        cppgd.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
        )
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, loss=lambda out: out)
    model.eval()
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    pts = torch.stack([x, E], dim=1)
    u = model(pts)
    assert u.shape == (2, 1)
    assert u.detach().numpy() == pytest.approx(
        cppgd.evaluate(pts).detach().numpy(), rel=1e-6
    )


def test_neurommodel_eval_forward_requires_coords(two_specs):
    cppgd = CPPGD(monom_specs=two_specs, n_modes_max=1, n_modes_ini=1)
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, loss=lambda out: out)
    model.eval()
    with pytest.raises(ValueError):
        model()


def test_neurommodel_assemble_delegates(two_specs):
    cppgd = CPPGD(monom_specs=two_specs, n_modes_max=1, n_modes_ini=1)
    domain = IntegrationDomain(cppgd.assemblies())
    model = NeuROMModel(FieldLayout(), cppgd, domain, loss=lambda out: out)
    x = torch.tensor([2.5, 5.0])
    E = torch.tensor([400.0, 700.0])
    # Pure delegation check on untouched monoms: both sides are exactly zero,
    # so a relative tolerance is meaningless here -- compare bit-for-bit.
    assert torch.allclose(model.assemble([x, E]), cppgd.assemble([x, E]))


def test_neurommodel_never_touches_cp_specific_attributes():
    """NeuROMModel must go through the TensorDecomposition contract only.

    Driven by a fake decomposition with no CP structure at all. If any
    CP-specific attribute access (``monoms``, ``factors``, ``n_active_modes``, ...)
    creeps into NeuROMModel, every CPPGD-backed test stays green and only this
    one fails -- with an AttributeError. All three public entry points are
    exercised for that reason: train forward, eval forward and assemble.
    """
    layout = FieldLayout()
    deco = _ConstantDecomposition()
    domain = IntegrationDomain(deco.assemblies())
    model = NeuROMModel(
        layout, deco, domain, loss=lambda out: out["dummy"].u.values.sum()
    )

    # Train forward: fills the layout through the domain. Independent expected
    # value: the fake's field is ones on 3 elements x 2 quadrature points,
    # interpolated to ones -> sum 6.0. Also proves the domain really
    # interpolated the field (an unfilled field would raise on `.u`).
    out = model()
    assert float(model.loss(out).detach()) == 6.0

    # Eval forward and assemble, on the same structure-free fake.
    model.eval()
    assert model([torch.zeros(3)]).shape == (3, 1)
    assert model.assemble([torch.zeros(3), torch.zeros(2)]).shape == (3, 2)
