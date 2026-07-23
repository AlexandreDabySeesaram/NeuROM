import pytest
import torch

from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearBar
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Connectivity
from neurom.fields import Field, TrainableField
from neurom.constraints import NoConstraint
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)

torch.set_default_dtype(torch.float32)


def _ctx(n=4):
    coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    conn = Connectivity(nodes, elements)
    positions = Field(name="x", connectivity=conn, values=coords)
    sf = LinearBar()
    mesh = Mesh(connectivity=conn, nodes_positions=positions)
    ctx = QuadratureContext(mesh, TwoPoints1D(), IsoparametricMapping1D(sf, mesh))
    return ctx, conn, sf


def _field(name, conn, n=4):
    return TrainableField(
        name=name,
        connectivity=conn,
        init_values=torch.ones(n, 1),
        constraint=NoConstraint(),
    )


def _layout(fields):
    layout = FieldLayout()
    for f in fields:
        layout.add(f)
    return layout


def test_active_defaults_true():
    ctx, conn, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("w", conn))
    assert bool(a.active) is True


def test_active_is_a_buffer():
    ctx, conn, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("w", conn))
    assert "active" in dict(a.named_buffers())


def test_inactive_assembly_is_not_interpolated():
    ctx, conn, sf = _ctx()
    fa = _field("wa", conn)
    fb = _field("wb", conn)
    a = QuadratureAssembly(ctx, sf, fa, active=True)
    b = QuadratureAssembly(ctx, sf, fb, active=False)
    domain = IntegrationDomain([a, b])
    layout = _layout([fa, fb])
    domain.interpolate_all(layout)
    assert layout[fa.name].u.shape[-1] == 1  # active -> interpolated
    with pytest.raises(RuntimeError):  # inactive -> not interpolated
        _ = layout[fb.name]


def test_activate_makes_next_interpolation_include_it():
    ctx, conn, sf = _ctx()
    fb = _field("wb", conn)
    b = QuadratureAssembly(ctx, sf, fb, active=False)
    domain = IntegrationDomain([b])
    layout = _layout([fb])
    b.activate()
    domain.interpolate_all(layout)
    assert layout[fb.name].u.shape[-1] == 1


def test_contexts_deduplicated_across_assemblies():
    ctx, conn, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("wa", conn))
    b = QuadratureAssembly(ctx, sf, _field("wb", conn))
    domain = IntegrationDomain([a, b])
    assert len(domain._contexts) == 1
    assert domain._contexts[0] is ctx
