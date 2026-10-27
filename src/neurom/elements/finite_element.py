"""Finite element: owns the DOF layout, so DOF counts/locations are deduced, not passed.

Shape glossary: ``n_dofs`` (= ``n_phys``) physical DOFs per element, ``n_ref`` reference basis
functions, ``dim_ref`` the reference/parametric dimension (1 bar, 2 triangle), ``gdim`` the
geometric (ambient) dimension, ``N_e`` elements, ``N_q`` quadrature/evaluation points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from neurom.meshes.topology import EntityDim

if TYPE_CHECKING:
    from neurom.dof_transformations.functionals import DofFunctional
    from neurom.shape_functions.shape_function import ShapeFunction

# DOF kinds (the functional defining each DOF).
VALUE = "value"  # nodal value
D1 = "d1"  # nodal first derivative (gradient)
FLUX = "flux"  # edge/face normal flux (RT0)
NORMAL_DERIVATIVE = "normal_derivative"  # Morley/HCT/Argyris (coupled)


@dataclass(frozen=True)
class DoF:
    """One degree of freedom in an element's declarative layout (what & where, mesh-independent).

    A pure databag: the stored payload is just ``(entity_dim, local_index, kind)``. The DOF's
    functional (what it *means*) is a **derived view** of ``kind`` -- see the :attr:`functional`
    property -- not a stored field, so a DOF is complete the moment it is constructed and no external
    code has to resolve anything.

    Attributes:
        entity_dim (EntityDim): Dimension of the cell sub-entity the DOF is attached to
            (``VERTEX``/``EDGE``/``FACE``/``VOLUME``).
        local_index (int): Ordinal of that entity within the reference cell (e.g. which of a
            triangle's three vertices or edges).
        kind (str): The DOF's label (``VALUE``/``D1``/``FLUX``/``NORMAL_DERIVATIVE``) -- the single
            stored payload. Routes the DOF (``locate`` / BCs / ``dof_kinds``) *and* selects its
            functional. Per-DOF variation within a kind is carried by the other fields that the
            (shared) functional reads -- e.g. a flux functional reads ``local_index`` for which edge.
    """

    entity_dim: EntityDim
    local_index: int
    kind: str

    def __post_init__(self):
        # Keep a typed EntityDim and an int index, however the DoF was built.
        object.__setattr__(self, "entity_dim", EntityDim(int(self.entity_dim)))
        object.__setattr__(self, "local_index", int(self.local_index))

    @property
    def functional(self) -> DofFunctional | None:
        """The DOF's functional, derived from ``kind`` (``None`` if the kind has no functional yet).

        A derived view, not a stored field: ``FUNCTIONALS[kind]`` returns the *same shared instance*
        on every access, so grouping DOFs by functional identity (in :class:`~neurom.dof_transformations.transformation.Transformation`
        and :meth:`FiniteElement.duality_matrix`) still batches same-kind DOFs together. A kind with
        no functional yet (``flux``/``normal_derivative``) returns ``None``; the consumers that need
        one (interpolation, the DOF transformation, the duality check) handle that explicitly.

        Returns:
            DofFunctional | None: The shared functional for ``kind``, or ``None``.
        """
        from neurom.dof_transformations.functionals import FUNCTIONALS

        return FUNCTIONALS.get(self.kind)


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
        nodal (bool): Whether the coded basis is dual to the DOFs (default ``True``). If ``True``,
            the construction-time duality check requires ``V == I``; set ``False`` for a non-nodal
            (e.g. hierarchical) basis, whose ``C = V^{-1}`` is folded into the tabulation later.

    Attributes:
        reference_basis (ShapeFunction): The reference basis.
        cell (ReferenceCell): The reference cell (``reference_basis.reference_cell``).
        dof_layout (tuple[DoF, ...]): The normalised DOF layout.
        value_shape (tuple[int, ...]): Field component shape.
        pullback (str): The push-forward label.
        nodal (bool): Whether the element is declared nodal.
    """

    def __init__(
        self,
        reference_basis: ShapeFunction,
        dof_layout: Sequence[DoF | tuple],
        value_shape: tuple[int, ...] = (),
        pullback: str = "identity",
        nodal: bool = True,
    ) -> None:
        self.reference_basis = reference_basis
        self.cell = reference_basis.reference_cell
        # Normalise each entry to a DoF (a raw triple is accepted).
        self.dof_layout = tuple(
            d if isinstance(d, DoF) else DoF(*d) for d in dof_layout
        )
        self.value_shape = tuple(value_shape)
        self.pullback = pullback
        self.nodal = nodal
        self._check_duality()

    def duality_matrix(self) -> torch.Tensor | None:
        r"""The duality matrix :math:`V_{ab} = \hat\ell_a(\hat\Phi_b)`, or ``None``.

        Each DOF's *reference* functional (geometry-free, local orientation) applied to the coded
        reference basis at the DOF's reference point. ``V == I`` means the basis is nodal-dual to
        the DOFs; a non-nodal element instead folds ``C = V^{-1}`` into its tabulation. Returns
        ``None`` when any DOF kind has no functional yet (duality is then unverifiable here).

        Returns:
            torch.Tensor | None: ``(n_dofs, n_ref)``, or ``None``.
        """
        if any(d.functional is None for d in self.dof_layout):
            return None

        # (n_dofs, dim_ref) -- reference point of each DOF
        xi = self.dof_nodes_xi()

        # Group the DOF row-indices by their functional (DOFs sharing one are batched)
        by_func: dict[DofFunctional, list[int]] = {}
        for i, d in enumerate(self.dof_layout):
            by_func.setdefault(d.functional, []).append(i)

        # Fill the duality matrix with the reference functionals
        V = torch.empty(self.n_dofs, self.n_ref, dtype=xi.dtype)  # (n_dofs, n_ref)
        for func, rows in by_func.items():
            V[rows, :] = func.reference_row(self, xi[rows])

        return V

    def _check_duality(self) -> None:
        """Construction-time unisolvence / nodality check.

        For a ``nodal`` element (the default), the duality matrix must be the identity: that means
        ``n_dofs == n_ref`` (unisolvent) and the coded basis really is dual to the DOFs. A mismatch
        -- a wrong basis or a wrong DOF definition -- is caught here rather than silently producing
        garbage downstream. (Non-nodal elements fold ``C = V^{-1}`` into the tabulation instead --
        a later step.)
        """
        V = self.duality_matrix()
        if V is None or not self.nodal:
            return
        if self.n_dofs != self.n_ref:
            raise ValueError(
                f"nodal element is not unisolvent: {self.n_dofs} DOFs but {self.n_ref} basis "
                f"functions (n_dofs must equal n_ref)."
            )
        if not torch.allclose(V, torch.eye(self.n_dofs, dtype=V.dtype), atol=1e-6):
            raise ValueError(
                "element declared nodal, but its basis is not dual to its DOFs "
                "(the duality matrix V = l_a(phi_b) is not the identity); check the basis "
                "or the DOF definitions, or declare the element non-nodal."
            )

    @property
    def dim_ref(self) -> int:
        r"""Reference (parametric) dimension: the number of reference coordinates :math:`\xi`.

        ``1`` for a bar, ``2`` for a triangle (``= cell.simplex.shape[1]``). This is **not** the
        vertex count (a triangle has 3 vertices, a bar 2) and **not** the geometric/ambient
        dimension ``gdim`` (a bar may live in 2-D: ``dim_ref = 1`` while ``gdim = 2``).
        """
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
    def dof_kinds(self) -> tuple[str, ...]:
        """The ``kind`` of each DOF, in layout order, shape ``(n_dofs,)``."""
        return tuple(d.kind for d in self.dof_layout)

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate the reference basis at ``xi`` (delegates to ``reference_basis``).

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Basis values, shape ``(N_e, N_q, n_ref)``.
        """
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
