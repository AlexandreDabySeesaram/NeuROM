"""Descriptors of one factor of a separated representation.

A separated representation ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` is described
by two kinds of object, and the split between them is the point of this module:

``FactorSpace``
    *Where* the k-th factor lives: a mesh, a mapping, a quadrature rule, and the
    ``QuadratureContext`` built from them. One per factor, **shared** by every
    field posed on it -- all the monoms of that factor, mode after mode, and any
    other field read on the same quadrature points (a load, a source). Sharing
    the context is what makes ``inner(f, u)`` meaningful, since both operands are
    then sampled at the very same points; it is also what lets
    ``IntegrationDomain`` update that geometry once per forward instead of once
    per monom.

``MonomSpec``
    *What* is posed on it: how a monom is interpolated, what constrains it, and
    what it starts from. It is a **blueprint, not a monom**: a decomposition with
    two factors takes two ``MonomSpec``, whatever its number of modes, and
    :class:`~neurom.decompositions.pgd.CPPGD` instantiates one
    ``TrainableField`` per (mode, spec) pair from it. The three monoms
    ``w_0^x, w_1^x, w_2^x`` all come out of the same ``MonomSpec``.

So a 2-factor, 3-mode CP-PGD is built from 2 ``FactorSpace`` and 2 ``MonomSpec``,
and holds 6 ``TrainableField`` over 2 shared contexts.
"""

from dataclasses import dataclass

import torch

from dataclasses import field as dataclass_field

from neurom.elements.finite_element import FiniteElement
from neurom.fields.field import Field
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.meshes.connectivity import Connectivity
from neurom.meshes.mesh import Mesh
from neurom.quadratures.quadrature_rule import QuadratureRule


@dataclass
class FactorSpace:
    """Discretised space one factor of a separated representation lives on.

    Holds the geometry and the integration rule -- everything shared by every
    field posed on that space, and nothing else. See the module docstring for
    how it pairs with :class:`MonomSpec`.

    The ``mesh`` is built by the caller and injected; the geometric mapping and the
    ``QuadratureContext`` are built here from it (the mapping's geometry basis comes from
    the mesh's coordinate element). The space carries no field shape function of its own:
    that belongs to the :class:`MonomSpec` posed on it, which allows a sub/super-parametric
    element whose field basis differs from the geometry basis.

    Attributes:
        name (str): Factor name, used as key in the separated interpolation
            output and in the monom field names.
        mesh (Mesh): Mesh of this space; also carries ``coordinates`` and the topology the
            fields are numbered over.
        quad (QuadratureRule): Quadrature rule for integration on this space.
        context (QuadratureContext): Built here, in ``__post_init__``, so it is
            a first-class attribute every field on this space can share.
    """

    name: str
    mesh: Mesh
    quad: QuadratureRule

    @property
    def connectivity(self) -> Connectivity:
        return self.mesh.connectivity

    @property
    def coordinates(self) -> Field:
        return self.mesh.coordinates

    def __post_init__(self):
        self.context = QuadratureContext(self.mesh, self.quad)


@dataclass
class MonomSpec:
    """Blueprint for the monoms of one factor: what is posed on a FactorSpace.

    Pairs a :class:`FactorSpace` with everything that belongs to the field
    rather than to the space -- how it is interpolated, what constrains it, and
    what it starts from.

    **One spec, many monoms.** This is not a monom ``w_m^k``: it is the recipe
    :class:`~neurom.decompositions.pgd.CPPGD` reads to build one
    ``TrainableField`` per mode on that factor, all sharing these ``bcs`` and
    seeded from these ``init_values``. Enriching the decomposition with a mode
    instantiates one more field per spec; it never creates a spec.

    Several ``MonomSpec`` may share one ``FactorSpace`` -- two differently
    constrained fields on a single mesh and quadrature, with the geometry
    computed once.

    Attributes:
        space (FactorSpace): The discretised space this factor lives on.
        element (FiniteElement): Finite element used to interpolate this factor's
            monoms; its DOF layout drives the (deduced) DOF numbering. Unrelated to
            the geometry shape function held by ``space.mapping``.
        bcs (Sequence[DirichletBC]): Essential boundary conditions shared by every
            monom built from this spec (empty for an unconstrained factor).
        init_values (torch.Tensor): Initial nodal values for each new monom,
            shape (n_nodes, dim). Its width is the monom's value dimension: 1
            for a scalar factor, >1 for the single vector-valued factor a CP
            decomposition admits.
    """

    space: FactorSpace
    element: FiniteElement
    init_values: torch.Tensor
    bcs: tuple = dataclass_field(default_factory=tuple)
