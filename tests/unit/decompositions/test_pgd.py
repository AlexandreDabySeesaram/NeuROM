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
