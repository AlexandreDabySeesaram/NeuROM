"""Finite element: owns the DOF layout, so DOF counts/locations are deduced, not passed."""

import math
from dataclasses import dataclass

import torch

from neurom.meshes.topology import EntityDim

# DOF kinds (the functional defining each DOF).
VALUE = "value"  # nodal value
D1 = "d1"  # nodal first derivative (gradient)
FLUX = "flux"  # edge/face normal flux (RT0)
NORMAL_DERIVATIVE = "normal_derivative"  # Morley/HCT/Argyris (coupled)


@dataclass(frozen=True)
class DoF:
    """One degree of freedom in an element's declarative layout (what & where, mesh-independent).

    Attributes:
        entity_dim (EntityDim): Dimension of the cell sub-entity the DOF is attached to
            (``VERTEX``/``EDGE``/``FACE``/``VOLUME``).
        local_index (int): Ordinal of that entity within the reference cell (e.g. which of a
            triangle's three vertices or edges).
        kind (str): The functional defining the DOF -- one of ``VALUE``, ``D1``, ``FLUX``,
            ``NORMAL_DERIVATIVE``.
    """

    entity_dim: EntityDim
    local_index: int
    kind: str


class FiniteElement:
    r"""A finite element = reference basis + a declarative DOF layout.

    A plain immutable object (not an ``nn.Module``): it holds only the reference basis, the
    declarative DOF layout and the value shape -- no geometry, no trainable state, no dtype-pinned
    tensors -- so a single shared instance (the library singletons) serves every mesh and dtype,
    and casting one model never mutates another's element.

    The ``dof_layout`` is the single source of truth for *what* and *where* the
    DOFs are; everything the user previously had to pass (DOFs-per-node, the field
    connectivity, the reference node coordinates) is derived from it by the
    ``DofMap``/``FunctionSpace``.

    Args:
        reference_basis (ShapeFunction): Evaluates the reference basis :math:`N(\xi)`;
            supplies the reference element (``reference_basis.reference_cell``).
        dof_layout (Sequence[DoF]): Ordered DOFs, in the same order as the columns of ``N``.
            Each entry is a :class:`DoF` (``(entity_dim, local_index, kind)`` triples are also
            accepted and coerced). E.g. the Hermite beam is ``[DoF(VERTEX,0,VALUE),
            DoF(VERTEX,0,D1), DoF(VERTEX,1,VALUE), DoF(VERTEX,1,D1)]``.
        value_shape (tuple[int, ...]): Component shape of the field value; ``()``
            scalar, ``(2,)`` a 2-vector. Orthogonal to ``dof_layout``.
        pullback (str): ``"identity"`` (scalar/Lagrange/Hermite) or ``"piola"`` (RT0).

    Attributes:
        reference_basis, cell, dof_layout, value_shape, pullback.
    """

    def __init__(
        self, reference_basis, dof_layout, value_shape=(), pullback="identity"
    ):
        self.reference_basis = reference_basis
        self.cell = reference_basis.reference_cell
        # DoF normalises its own fields (__post_init__); a raw triple is accepted as a shorthand.
        self.dof_layout = tuple(
            d if isinstance(d, DoF) else DoF(*d) for d in dof_layout
        )
        self.value_shape = tuple(value_shape)
        self.pullback = pullback

    @property
    def dim_ref(self) -> int:
        """Reference (parametric) dimension: 1 for a bar, 2 for a triangle."""
        return self.cell.dim_ref

    @property
    def n_dofs(self) -> int:
        """Number of physical DOFs per element (``n_phys`` = length of ``dof_layout``)."""
        return len(self.dof_layout)

    @property
    def n_ref(self) -> int:
        r"""Number of reference basis functions = number of columns of :math:`N(\xi)`.

        Obtained by evaluating the basis at a single probe point -- the first reference vertex,
        shaped ``(1, 1, dim_ref)`` to match ``N``'s ``(N_e, N_q, dim_ref)`` input -- and reading
        the trailing axis. May differ from ``n_dofs`` (e.g. a hierarchical/bubble basis).
        """
        probe = self.cell.simplex[:1].unsqueeze(0)  # (1, 1, dim_ref), first vertex
        return self.reference_basis.N(probe).shape[-1]

    @property
    def n_components(self) -> int:
        """Number of scalar value components = product of ``value_shape`` (1 when scalar)."""
        return math.prod(self.value_shape)

    @property
    def dof_kinds(self) -> tuple:
        """The ``kind`` of each DOF, in layout order."""
        return tuple(d.kind for d in self.dof_layout)

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate the reference basis at ``xi`` (delegates to ``reference_basis``)."""
        return self.reference_basis.N(xi)

    def dof_nodes_xi(self) -> torch.Tensor:
        """Reference coordinate carrying each DOF, deduced from ``dof_layout`` + ``cell``.

        A vertex DOF sits at that vertex; an edge DOF at the edge midpoint; a cell DOF at
        the cell centroid. (Face DOF coordinates are added later.)

        Returns:
            torch.Tensor: ``(n_dofs, dim_ref)``.
        """
        verts = self.cell.simplex  # (n_verts, dim_ref)
        out = []
        for d in self.dof_layout:
            if d.entity_dim == EntityDim.VERTEX:
                out.append(verts[d.local_index])
            elif d.entity_dim == EntityDim.EDGE and self.dim_ref >= 2:
                # a true sub-edge (triangle and up): midpoint of its two local vertices
                edge = self.cell.local_entities(EntityDim.EDGE)[
                    d.local_index
                ]  # (2,) local vertices
                out.append(verts[edge].mean(dim=0))
            elif d.entity_dim == self.dim_ref:
                # the top-dimensional cell itself -- incl. a bar, whose sole edge IS the cell,
                # so its centroid coincides with the edge midpoint -> centroid
                out.append(verts.mean(dim=0))
            else:
                raise NotImplementedError(
                    "face DOF reference coordinates are not built yet"
                )
        return torch.stack(out)
