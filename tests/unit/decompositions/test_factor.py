import torch

from neurom.decompositions import MonomSpec
from neurom.constraints import Dirichlet, NoConstraint
from neurom.interpolation import IntegrationDomain, QuadratureAssembly
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.meshes import Mesh
from neurom.shape_functions import LinearBar
from neurom.fields import TrainableField

torch.set_default_dtype(torch.float32)


def test_factor_space_wires_mesh_context_and_connectivity(make_factor_space):
    """A FactorSpace owns one mesh and one context, all sharing the same objects.

    Every identity here is an invariant CPPGD relies on: the monoms bind to
    ``space.connectivity``, so a copy anywhere in this chain would silently
    detach them from the mesh they are supposed to live on.
    """
    space = make_factor_space()
    assert space.name == "space"
    assert isinstance(space.mesh, Mesh)
    assert isinstance(space.context, QuadratureContext)
    assert space.connectivity is space.nodes_positions.connectivity
    assert space.mesh.connectivity is space.connectivity
    assert space.mesh.nodes_positions is space.nodes_positions
    assert space.connectivity.n_nodes == 5


def test_two_specs_have_distinct_contexts(two_specs):
    space, para = two_specs
    assert space.space.context is not para.space.context
    assert space.space.mesh is not para.space.mesh


def test_monom_spec_holds_the_field_side_only(make_factor_space):
    """``sf``, ``constraint`` and ``init_values`` belong to the spec, not the space."""
    space = make_factor_space()
    factor = MonomSpec(
        space=space,
        sf=LinearBar(),
        constraint=NoConstraint(),
        init_values=torch.zeros(5, 1),
    )
    assert factor.space is space
    assert not hasattr(space, "sf")
    assert not hasattr(space, "constraint")
    assert not hasattr(space, "init_values")


def test_two_specs_share_one_space_and_its_context(make_factor_space):
    """Two differently-constrained factors can live on a single FactorSpace.

    This is what the FactorSpace/MonomSpec split buys: the geometry and the
    quadrature are built once and shared, while each spec keeps its own
    constraint and seed. The shared context is an identity, so
    ``IntegrationDomain`` dedups it down to a single geometry update per
    forward.
    """
    space = make_factor_space(n=5)
    free = MonomSpec(
        space=space,
        sf=LinearBar(),
        constraint=NoConstraint(),
        init_values=torch.zeros(5, 1),
    )
    clamped = MonomSpec(
        space=space,
        sf=LinearBar(),
        constraint=Dirichlet(nodes=[0], values_imposed=torch.zeros(1, 1)),
        init_values=torch.ones(5, 1),
    )

    assert free.space is clamped.space
    assert free.space.context is clamped.space.context
    assert free.constraint is not clamped.constraint

    assemblies = [
        QuadratureAssembly(
            f.space.context,
            f.sf,
            TrainableField(
                name=name,
                connectivity=f.space.connectivity,
                init_values=f.init_values,
                constraint=f.constraint,
            ),
        )
        for name, f in (("free", free), ("clamped", clamped))
    ]
    domain = IntegrationDomain(assemblies)
    assert len(domain.assemblies) == 2
    assert len(domain._contexts) == 1
