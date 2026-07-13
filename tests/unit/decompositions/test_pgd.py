import pytest
import torch

from neurom.decompositions import Axis, CPPGD
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import NoConstraint
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.decompositions import TensorDecomposition
from neurom.field_layout import FieldLayout
from neurom.decompositions import PGDFEMModel
from neurom.integrate import integrate
from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult

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


def test_assemble_sums_two_modes_matching_manual_outer_products():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=2)

    # NoConstraint on both axes -> nodal values are the full field. Set them
    # to distinct known vectors for each of the two modes.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)  # S0 at 5 space nodes
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)  # g0 at 4 E nodes
        )
        model.monoms[1][0].values_reduced.copy_(
            torch.tensor([1.0, -1.0, 2.0, 0.5, -0.5]).unsqueeze(-1)  # S1
        )
        model.monoms[1][1].values_reduced.copy_(
            torch.tensor([-2.0, 1.0, 0.0, 3.0]).unsqueeze(-1)  # g1
        )

    x = torch.tensor([2.5, 5.0])          # inside space domain [0, 10]
    E = torch.tensor([400.0, 700.0])      # inside E domain [100, 1000]
    u = model.assemble([x, E])

    assert u.shape == (2, 2)

    from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

    pwi_s0 = PointWiseInterpolator(
        model._meshes[0], axes[0].sf, model.monoms[0][0], axes[0].mapping
    )
    pwi_e0 = PointWiseInterpolator(
        model._meshes[1], axes[1].sf, model.monoms[0][1], axes[1].mapping
    )
    pwi_s1 = PointWiseInterpolator(
        model._meshes[0], axes[0].sf, model.monoms[1][0], axes[0].mapping
    )
    pwi_e1 = PointWiseInterpolator(
        model._meshes[1], axes[1].sf, model.monoms[1][1], axes[1].mapping
    )
    s0 = pwi_s0.at_position(x).reshape(-1)
    g0 = pwi_e0.at_position(E).reshape(-1)
    s1 = pwi_s1.at_position(x).reshape(-1)
    g1 = pwi_e1.at_position(E).reshape(-1)
    expected = torch.outer(s0, g0) + torch.outer(s1, g1)
    assert torch.allclose(u, expected, atol=1e-5)


def test_add_mode_activates_new_without_freezing_previous():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=2, n_modes_ini=1)

    # Dirty the (frozen) mode-1 monoms so we can check zero-out.
    with torch.no_grad():
        for f in model.monoms[1]:
            f.values_reduced.add_(7.0)

    new = model.add_mode()

    assert new == 1
    assert int(model.n_modes_truncated) == 2
    # Previous mode left untouched (still active), new mode active
    assert all(f.values_reduced.requires_grad for f in model.monoms[0])
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


def test_add_mode_to_optimizer_explicit_index():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=3, n_modes_ini=1)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    model.add_mode()
    model.add_mode()
    n_before = sum(len(g["params"]) for g in optim.param_groups)

    # Explicit index adds that mode's monoms.
    model.add_mode_to_optimizer(optim, m=1)
    assert sum(len(g["params"]) for g in optim.param_groups) == n_before + 2

    # Negative index resolves against the active modes.
    model.add_mode_to_optimizer(optim, m=-1)
    assert sum(len(g["params"]) for g in optim.param_groups) == n_before + 4


def test_add_mode_to_optimizer_out_of_range_raises():
    axes = make_two_axes()
    model = CPPGD(axes=axes, n_modes_max=3, n_modes_ini=1)
    optim = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    # Only mode 0 is active.
    with pytest.raises(IndexError):
        model.add_mode_to_optimizer(optim, m=1)
    with pytest.raises(IndexError):
        model.add_mode_to_optimizer(optim, m=-2)


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
