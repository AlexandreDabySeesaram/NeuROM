import pytest
import torch

from neurom.quadratures import TwoPoints1D
from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.elements import P1_BAR, VectorElement
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)

torch.set_default_dtype(torch.float32)


def _ctx(n=4):
    coords = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, coords))
    ctx = QuadratureContext(mesh, TwoPoints1D())
    return ctx, FunctionSpace(mesh.topology, P1_BAR)


def _field(name, space, n=4):
    # Empty bcs compile to NoConstraint (every DOF free), as in the original test.
    return TrainableField(space, torch.ones(n, 1), bcs=[], name=name)


def _layout(fields):
    layout = FieldLayout()
    for f in fields:
        layout.add(f)
    return layout


def test_static_assemblies_are_interpolated():
    ctx, space = _ctx()
    fa = _field("wa", space)
    a = QuadratureAssembly(ctx, fa)
    domain = IntegrationDomain([a])
    layout = _layout([fa])
    domain.interpolate_all(layout)
    assert layout[fa.name].u.shape[-1] == 1


def test_per_call_assemblies_are_interpolated_on_top_of_the_static_ones():
    """The seam a growing decomposition uses: its assemblies are not stored here.

    ``b`` is unknown to the domain and only handed over at call time, the way
    ``NeuROMModel.forward`` passes ``decomposition.assemblies()``.
    """
    ctx, space = _ctx()
    fa, fb = _field("wa", space), _field("wb", space)
    a = QuadratureAssembly(ctx, fa)
    b = QuadratureAssembly(ctx, fb)
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
    ctx_static, space = _ctx()
    ctx_dynamic, space_d = _ctx(n=5)
    a = QuadratureAssembly(ctx_static, _field("wa", space))
    b = QuadratureAssembly(ctx_dynamic, _field("wb", space_d, n=5))
    domain = IntegrationDomain([a])
    assert ctx_dynamic not in domain._contexts

    updated = []
    for ctx in (ctx_static, ctx_dynamic):
        ctx.update = (lambda c: lambda: updated.append(c))(ctx)

    domain.update_contexts([b])
    assert updated == [ctx_static, ctx_dynamic]


def test_contexts_deduplicated_across_assemblies():
    ctx, space = _ctx()
    a = QuadratureAssembly(ctx, _field("wa", space))
    b = QuadratureAssembly(ctx, _field("wb", space))
    domain = IntegrationDomain([a, b])
    assert len(domain._contexts) == 1
    assert domain._contexts[0] is ctx
