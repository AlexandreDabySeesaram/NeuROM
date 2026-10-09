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
        value (float | Sequence[float]): Imposed value(s). A scalar applies to every fixed
            component; a sequence must match the fixed components (``components`` if given,
            else all ``n_components``).
    """

    def __init__(
        self,
        entity_dim: int | EntityDim,
        entities,
        kinds=None,
        components=None,
        value=0.0,
    ):
        self.entity_dim = entity_dim
        self.entities = entities
        self.kinds = kinds
        self.components = components
        self.value = value

    def dofs(self, dof_map: DofMap) -> torch.Tensor:
        """Resolve to global scalar-DOF indices via the DOF map."""
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
        self.dof_map = DofMap(topology, element, periodic=periodic)
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

    def locate(self, entity_dim: int | EntityDim, entities, kinds=None) -> torch.Tensor:
        """Global scalar-DOF indices for an ``(entity, kind)`` selection (for BCs / init)."""
        return self.dof_map.locate(entity_dim, entities, kinds)

    def compile_constraint(self, bcs, n_comp: int):
        """Compile boundary conditions into a per-component free/fixed split.

        Resolves each BC against the DOF map and fills a ``(n_scalar_dofs, n_comp)`` grid of
        imposed values together with a boolean mask of which component slots are *free*. A
        whole-DOF clamp and a per-component roller are both just patterns in the mask; "no
        BC" leaves every slot free. The field (:class:`~neurom.fields.TrainableField`) stores
        the free slots as its parameter and scatters on expansion.

        Args:
            bcs (Sequence[DirichletBC]): The essential boundary conditions.
            n_comp (int): The field's component count.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(free_mask, imposed)``, each
            ``(n_scalar_dofs, n_comp)`` -- ``free_mask`` is ``True`` where the slot is
            trainable, ``imposed`` holds the fixed values (``0`` on free slots).
        """
        n = self.n_scalar_dofs
        fixed = torch.zeros(n, n_comp, dtype=torch.bool)
        imposed = torch.zeros(n, n_comp, dtype=torch.get_default_dtype())
        for bc in bcs:
            comps, vals = bc.component_values(n_comp)
            for d in bc.dofs(self.dof_map).tolist():
                for c, v in zip(comps, vals):
                    fixed[d, c] = True
                    imposed[d, c] = v
        return ~fixed, imposed
