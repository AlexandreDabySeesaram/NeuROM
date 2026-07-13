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
