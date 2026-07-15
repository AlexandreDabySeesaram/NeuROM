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
from neurom.integrate import integrate
from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult
from neurom.neurom_model import NeuROMModel

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


def test_interpolate_separated_is_removed():
    model = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    assert not hasattr(model, "interpolate_separated")


def test_separated_view_is_removed():
    model = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    assert not hasattr(model, "separated_view")


def test_pgdfemmodel_export_is_removed():
    import neurom.decompositions as d
    assert not hasattr(d, "PGDFEMModel")


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


class _ConstantDecomposition(TensorDecomposition):
    """Minimal fake decomposition with NO CP structure: registers one fixed
    Field and fills it with a constant result. Proves NeuROMModel is generic."""

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

    def evaluate(self, coords):
        return torch.ones(coords[0].reshape(-1).shape[0], 1)

    def assemble(self, coords):
        return torch.ones(*[c.reshape(-1).shape[0] for c in coords])


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
    u = model.evaluate(torch.stack([x, E], dim=1))   # matched, (2, 1)
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

    u = model.evaluate(torch.stack([x, E], dim=1))
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
    pts = torch.stack([x, E], dim=1)
    u = model(pts)
    assert u.shape == (2, 1)
    assert torch.allclose(u, cppgd.evaluate(pts), atol=1e-6)


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
