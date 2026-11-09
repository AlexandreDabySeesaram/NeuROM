"""FunctionSpace: ``(topology, element)`` -> global DOF numbering and BC compilation.

A space is the discrete function space over a mesh topology for a given finite element. Its
sole job is the DOF layout: how many scalar DOFs there are, which cell touches which global
DOF (the DOF map), where a named set of DOFs lives (:meth:`locate`), and how a set of
boundary conditions reduces to a free/fixed split (:meth:`compile_constraint`). It makes no
fields, interpolates nothing and assembles nothing -- those live on the fields and the
integration layer.
"""

import torch
import torch.nn as nn

from neurom.meshes.topology import Topology, EntityDim
from neurom.function_space.dof_map import DofMap
from neurom.meshes.connectivity import Connectivity


class DirichletBC:
    """Essential boundary condition, declared by *meaning* rather than by DOF index.

    By default every value component of the selected DOFs is fixed (a clamp). For a vector
    field pass a per-component ``value`` (e.g. ``[0.0, -1.0]``), and/or ``components`` to fix
    only some components and leave the rest free (a roller: ``components=[0]`` fixes
    :math:`u_x` only).

    Args:
        entity_dim (int | EntityDim): Entity dimension (``EntityDim.VERTEX`` vertices,
            ``topology.cell_dim`` cells).
        entities (Sequence[int]): Global entity indices (e.g. the end vertices).
        kinds (Sequence[str], optional): DOF kinds to fix; defaults to every kind on that
            entity dimension.
        components (Sequence[int], optional): Value-component indices to fix; defaults to all
            components of the field.
        value (float | Sequence[float] | torch.Tensor): Imposed value(s). A scalar applies to
            every fixed component; a sequence must match the fixed components (``components`` if
            given, else all ``n_components``). A **per-DOF tensor** ``(n_scalar_dofs, n_components)``
            imposes function-valued data: evaluate ``g = interpolate(space, mesh, f)`` (the
            mesh-explicit path) and pass ``g`` -- only its selected DOFs are read, so the space
            stays geometry-free. A *callable* ``value`` is rejected (resolve it via ``interpolate``
            first).
        closure (bool): If ``True``, also fix the DOFs of the selected entities' sub-entities
            (an edge's vertices) -- so a P2 clamp built from boundary *edges* also clamps the
            boundary *vertices*. Defaults to ``False`` (only the entities' own DOFs).
    """

    def __init__(
        self,
        entity_dim: int | EntityDim,
        entities,
        kinds=None,
        components=None,
        value=0.0,
        closure=False,
    ):
        self.entity_dim = entity_dim
        self.entities = entities
        self.kinds = kinds
        self.components = components
        self.value = value
        self.closure = closure

    def dofs(self, dof_map: DofMap) -> torch.Tensor:
        """Resolve to global scalar-DOF indices via the DOF map (no closure; see FunctionSpace.locate)."""
        return dof_map.locate(self.entity_dim, self.entities, self.kinds)

    def component_values(self, n_components: int):
        """Return ``(components, values)``: the components this BC fixes and each value.

        Args:
            n_components (int): The field's component count.

        Returns:
            tuple[list[int], list[float]]: Aligned component indices and imposed values.

        Raises:
            ValueError: If a sequence ``value`` does not match the fixed component count.
        """
        comps = (
            list(range(n_components))
            if self.components is None
            else list(self.components)
        )
        if isinstance(self.value, (int, float)):
            vals = [float(self.value)] * len(comps)
        else:
            vals = [float(v) for v in self.value]
            if len(vals) != len(comps):
                raise ValueError(
                    f"DirichletBC value has {len(vals)} entries but fixes "
                    f"{len(comps)} component(s) {comps}."
                )
        return comps, vals


class FunctionSpace(nn.Module):
    """A finite-element function space over a topology: DOF numbering + BC compilation.

    A function space is purely topological -- ``(topology, element)``. It carries no
    geometry; building the FE interpolant of a function onto its DOFs
    (:func:`neurom.interpolation.interpolate`) takes the mesh explicitly, since that is where
    the coordinates live.

    Args:
        topology (Topology): The topology to number DOFs over.
        element (FiniteElement): The element; its ``dof_layout`` drives the DOF map.
        periodic (Sequence[tuple[int, int]], optional): Vertex pairs to identify
            topologically (periodic BCs live here, not in :class:`DirichletBC`).

    Attributes:
        topology (Topology): The mesh topology DOFs are numbered over.
        element (FiniteElement): The finite element.
        dof_map (DofMap): The global DOF numbering.
    """

    def __init__(self, topology: Topology, element, periodic=None):
        super().__init__()
        self.topology = topology
        self.element = element
        self.dof_map = DofMap(topology, element.dof_layout, periodic=periodic)
        self._connectivity = Connectivity(
            torch.arange(self.dof_map.n_scalar_dofs), self.dof_map.cell_dofs
        )

    @property
    def n_scalar_dofs(self) -> int:
        """Number of scalar DOFs (per component)."""
        return self.dof_map.n_scalar_dofs

    @property
    def connectivity(self) -> Connectivity:
        """The DOF connectivity ``(cell -> global scalar DOFs)``."""
        return self._connectivity

    def locate(
        self, entity_dim: int | EntityDim, entities, kinds=None, closure=False
    ) -> torch.Tensor:
        """Global scalar-DOF indices for an ``(entity, kind)`` selection (for BCs / init).

        With ``closure=True`` the result also includes the DOFs of the selected entities'
        sub-entities -- currently an edge's two vertices -- so a selection of boundary *edges*
        also returns the boundary *vertex* DOFs. Needs the topology (edge -> vertices), which is
        why closure lives here rather than on :class:`DofMap`.

        Args:
            entity_dim (int | EntityDim): Dimension of the entities.
            entities (Sequence[int]): Global entity indices.
            kinds (Sequence[str], optional): DOF kinds to select; defaults to all on that dimension.
            closure (bool): Also include sub-entity DOFs (edge -> its vertices).

        Returns:
            torch.Tensor: Sorted global scalar-DOF indices.
        """
        dofs = self.dof_map.locate(entity_dim, entities, kinds)
        if not closure or entity_dim == EntityDim.VERTEX:
            return dofs
        if entity_dim == EntityDim.EDGE:
            edges = torch.as_tensor(list(entities), dtype=torch.int64)
            verts = torch.unique(self.topology.edge_vertices[edges].reshape(-1))
            vdofs = self.dof_map.locate(EntityDim.VERTEX, verts.tolist(), kinds)
            return torch.unique(torch.cat([dofs, vdofs]))
        raise NotImplementedError(
            f"closure is implemented for edges (-> vertices); got entity_dim {int(entity_dim)}."
        )

    def compile_constraint(self, bcs, n_comp: int):
        """Compile boundary conditions into a per-component free/fixed split.

        Resolves each BC against the DOF map and fills a ``(n_scalar_dofs, n_comp)`` grid of
        imposed values together with a boolean mask of which component slots are *free*. A
        whole-DOF clamp and a per-component roller are both just patterns in the mask; "no
        BC" leaves every slot free. The field (:class:`~neurom.fields.TrainableField`) stores
        the free slots as its parameter and scatters on expansion.

        **Geometry-free by design:** this method never sees node positions. A BC value may be a
        constant (scalar / per-component) or a **per-DOF tensor** -- the latter is how function-
        valued boundary data is imposed: evaluate ``g = interpolate(space, mesh, f)`` (the
        mesh-explicit interpolation path) and pass ``g`` as the BC ``value``; only its boundary
        DOFs are read. Re-evaluating ``g`` when the geometry moves (r-adaptivity) is then an
        explicit choice in the geometry layer, never hidden in the space.

        Args:
            bcs (Sequence[DirichletBC]): The essential boundary conditions.
            n_comp (int): The field's component count.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(free_mask, imposed)``, each
            ``(n_scalar_dofs, n_comp)`` -- ``free_mask`` is ``True`` where the slot is
            trainable, ``imposed`` holds the fixed values (``0`` on free slots).

        Raises:
            TypeError: If a BC ``value`` is a callable -- resolve it via
                :func:`neurom.interpolation.interpolate` and pass the resulting per-DOF tensor
                (keeping the space geometry-free).
        """
        n = self.n_scalar_dofs
        fixed = torch.zeros(n, n_comp, dtype=torch.bool)
        imposed = torch.zeros(n, n_comp, dtype=torch.get_default_dtype())
        for bc in bcs:
            dofs = self.locate(bc.entity_dim, bc.entities, bc.kinds, closure=bc.closure)
            if dofs.numel() == 0:
                continue
            comps = (
                list(range(n_comp)) if bc.components is None else list(bc.components)
            )
            comps_t = torch.tensor(comps, dtype=torch.int64)
            rows = dofs.reshape(-1, 1)
            fixed[rows, comps_t] = True
            if callable(bc.value):
                raise TypeError(
                    "DirichletBC value is a callable; FunctionSpace is geometry-free. Resolve it "
                    "with g = interpolate(space, mesh, f) and pass g (a per-DOF tensor) as value."
                )
            if torch.is_tensor(bc.value):
                imposed[rows, comps_t] = bc.value.to(imposed.dtype)[dofs][:, comps_t]
            else:
                _, vals = bc.component_values(n_comp)
                imposed[rows, comps_t] = torch.tensor(vals, dtype=imposed.dtype)
        return ~fixed, imposed
