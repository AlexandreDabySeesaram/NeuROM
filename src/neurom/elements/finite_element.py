"""Finite element: owns the DOF layout, so DOF counts/locations are deduced, not passed."""

import torch
import torch.nn as nn

from neurom.meshes.topology import EntityDim

# DOF kinds (the functional defining each DOF).
VALUE = "value"  # nodal value
D1 = "d1"  # nodal first derivative (gradient)
FLUX = "flux"  # edge/face normal flux (RT0)
NORMAL_DERIVATIVE = "normal_derivative"  # Morley/HCT/Argyris (coupled)


class FiniteElement(nn.Module):
    """A finite element = reference basis + a declarative DOF layout.

    The ``dof_layout`` is the single source of truth for *what* and *where* the
    DOFs are; everything the user previously had to pass (DOFs-per-node, the field
    connectivity, the reference node coordinates) is derived from it by the
    ``DofMap``/``FunctionSpace``.

    Args:
        reference_basis (ShapeFunction): Evaluates the reference basis ``N(ξ)``;
            supplies the reference element (``reference_basis.reference_element``).
        dof_layout (Sequence): Ordered DOFs as ``(entity_dim, local_entity_index, kind)``
            triples, in the same order as the columns of ``N``. ``entity_dim`` is an
            :class:`~neurom.meshes.EntityDim` (``VERTEX``/``EDGE``/``FACE``/``VOLUME``);
            ``local_entity_index`` is the entity's local ordinal within the cell (e.g. which
            of a triangle's three vertices). E.g. the Hermite beam is
            ``[(VERTEX,0,VALUE),(VERTEX,0,D1),(VERTEX,1,VALUE),(VERTEX,1,D1)]``.
        value_shape (tuple[int, ...]): Component shape of the field value; ``()``
            scalar, ``(2,)`` a 2-vector. Orthogonal to ``dof_layout``.
        pullback (str): ``"identity"`` (scalar/Lagrange/Hermite) or ``"piola"`` (RT0).

    Attributes:
        reference_basis, cell, dof_layout, value_shape, pullback.
    """

    def __init__(
        self, reference_basis, dof_layout, value_shape=(), pullback="identity"
    ):
        super().__init__()
        self.reference_basis = reference_basis
        self.cell = reference_basis.reference_element
        # Normalise so the layout always carries a typed EntityDim, however it was declared.
        self.dof_layout = tuple(
            (EntityDim(int(entity_dim)), int(local_entity_index), kind)
            for entity_dim, local_entity_index, kind in dof_layout
        )
        self.value_shape = tuple(value_shape)
        self.pullback = pullback

    @property
    def dim_ref(self) -> int:
        """Reference (parametric) dimension: 1 for a bar, 2 for a triangle."""
        return self.cell.simplex.shape[1]

    @property
    def n_dofs(self) -> int:
        """Number of physical DOFs per element (``n_phys`` = length of ``dof_layout``)."""
        return len(self.dof_layout)

    @property
    def n_ref(self) -> int:
        """Number of reference basis functions (``n_ref``); may differ from ``n_dofs``."""
        probe = self.cell.simplex[:1].unsqueeze(0)  # (1, 1, dim_ref)
        return self.reference_basis.N(probe).shape[-1]

    @property
    def n_components(self) -> int:
        """Number of value components (product of ``value_shape``; 1 if scalar)."""
        n = 1
        for s in self.value_shape:
            n *= s
        return n

    @property
    def dof_kinds(self) -> tuple:
        """The ``kind`` of each DOF, in layout order."""
        return tuple(kind for (_, _, kind) in self.dof_layout)

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
        for entity_dim, local_entity_index, _ in self.dof_layout:
            if entity_dim == EntityDim.VERTEX:
                out.append(verts[local_entity_index])
            elif entity_dim == self.dim_ref:  # cell DOF -> centroid
                out.append(verts.mean(dim=0))
            elif entity_dim == EntityDim.EDGE:  # edge DOF -> reference edge midpoint
                edge = self.cell.local_edges[local_entity_index]  # (2,) local vertices
                out.append(verts[edge].mean(dim=0))
            else:
                raise NotImplementedError(
                    "face DOF reference coordinates are not built yet"
                )
        return torch.stack(out)
