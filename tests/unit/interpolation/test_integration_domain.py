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


def test_static_assemblies_are_interpolated():
    ctx, conn, sf = _ctx()
    fa = _field("wa", conn)
    a = QuadratureAssembly(ctx, sf, fa)
    domain = IntegrationDomain([a])
    layout = _layout([fa])
    domain.interpolate_all(layout)
    assert layout[fa.name].u.shape[-1] == 1


def test_per_call_assemblies_are_interpolated_on_top_of_the_static_ones():
    """The seam a growing decomposition uses: its assemblies are not stored here.

    ``b`` is unknown to the domain and only handed over at call time, the way
    ``NeuROMModel.forward`` passes ``decomposition.assemblies()``.
    """
    ctx, conn, sf = _ctx()
    fa, fb = _field("wa", conn), _field("wb", conn)
    a = QuadratureAssembly(ctx, sf, fa)
    b = QuadratureAssembly(ctx, sf, fb)
    domain = IntegrationDomain([a])
    layout = _layout([fa, fb])

    domain.interpolate_all(layout)
    with pytest.raises(RuntimeError):  # b was not passed -> not interpolated
        _ = layout[fb.name]

    domain.interpolate_all(layout, [b])
    assert layout[fa.name].u.shape[-1] == 1
    assert layout[fb.name].u.shape[-1] == 1


def test_update_contexts_reaches_a_per_call_only_context():
    """A context the domain never stored still gets refreshed when passed.

    Without this, a trainable mesh on a factor no static assembly touches would
    silently keep a stale geometry.
    """
    ctx_static, conn, sf = _ctx()
    ctx_dynamic, conn_d, sf_d = _ctx(n=5)
    a = QuadratureAssembly(ctx_static, sf, _field("wa", conn))
    b = QuadratureAssembly(ctx_dynamic, sf_d, _field("wb", conn_d, n=5))
    domain = IntegrationDomain([a])
    assert ctx_dynamic not in domain._contexts

    updated = []
    for ctx in (ctx_static, ctx_dynamic):
        ctx.update = (lambda c: lambda: updated.append(c))(ctx)

    domain.update_contexts([b])
    assert updated == [ctx_static, ctx_dynamic]


def test_contexts_deduplicated_across_assemblies():
    ctx, conn, sf = _ctx()
    a = QuadratureAssembly(ctx, sf, _field("wa", conn))
    b = QuadratureAssembly(ctx, sf, _field("wb", conn))
    domain = IntegrationDomain([a, b])
    assert len(domain._contexts) == 1
    assert domain._contexts[0] is ctx
