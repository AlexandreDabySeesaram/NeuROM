"""Global DOF numbering deduced from (topology, element) -- no user-supplied counts."""

import torch
import torch.nn as nn

from neurom.meshes.topology import EntityDim


class DofMap(nn.Module):
    """Assigns global DOFs from an element's ``dof_layout`` over a topology.

    Each entity of dimension ``d`` owns ``dofs_per_entity[d]`` consecutive scalar
    DOFs; a cell's local DOF ``i`` -- declared on ``(entity_dim,
    local_entity_index, kind)`` -- maps to the global DOF of that entity plus the
    DOF's rank within it.
    Shared entities (later: edges) therefore share DOFs automatically. Value
    *components* are a trailing axis on the field values, not part of this scalar
    numbering, so per-component BCs come from the constraint mask.

    Periodicity is a *topological identification*, not a constraint: a vertex pair
    ``(a, b)`` in ``periodic`` makes ``b``'s DOFs the *same* global DOFs as ``a``'s
    (matched by rank), so ``u(a) = u(b)`` holds by construction -- one shared
    parameter, gradients flowing to it -- the natural home for periodic BCs in the
    entity layer (orthogonal to the essential ``DirichletBC``).

    Numbering is purely combinatorial -- it needs only *what* DOFs exist and where they attach
    (``dof_layout``) and *how* entities are shared across cells (``topology``). It evaluates no basis,
    so it takes the element's ``dof_layout`` rather than the whole ``FiniteElement``.

    Args:
        topology (Topology): The mesh topology (used at construction for entity sharing/counts).
        dof_layout (Sequence[DoF]): The element's ordered DOF declarations (``element.dof_layout``).
        periodic (Sequence[tuple[int, int]], optional): Vertex pairs ``(a, b)`` to
            identify (``b``'s DOFs collapse onto ``a``'s). Defaults to none.

    Attributes:
        cell_dofs (torch.Tensor): ``(n_cells, n_local_dofs)`` scalar DOF indices
            (the field connectivity), a registered buffer.
        n_scalar_dofs (int): Total scalar DOFs (after any periodic identification).
    """

    def __init__(self, topology, dof_layout, periodic=None):
        super().__init__()
        layout = dof_layout

        dims_present = sorted({dof.entity_dim for dof in layout})

        # DOFs per entity and the ordered kinds on one entity, per dimension.
        self._dofs_per_entity: dict[int, int] = {}
        self._kinds_per_entity: dict[int, list] = {}
        for d in dims_present:
            entries = [
                (dof.local_index, dof.kind) for dof in layout if dof.entity_dim == d
            ]
            lei0 = min(lei for lei, _ in entries)
            kinds0 = [kind for lei, kind in entries if lei == lei0]
            self._dofs_per_entity[d] = len(kinds0)
            self._kinds_per_entity[d] = kinds0

        # Base offset of each dimension's DOF block, and the global total.
        # Order block by block based on DoF dimension: vertices first, then edges, etc.
        self._offset: dict[int, int] = {}
        total = 0
        for d in dims_present:
            self._offset[d] = total
            total += topology.n_entities(d) * self._dofs_per_entity[d]

        # Rank of each local DOF within its (entity_dim, local_index) group.
        # For case of having multiple type of DoFs on the same index, e.g. Hermite
        rank, seen = [], {}
        for dof in layout:
            key = (dof.entity_dim, dof.local_index)
            r = seen.get(key, 0)
            rank.append(r)
            seen[key] = r + 1

        # cell_dofs[e, i] = offset[d] + global_entity(e, i) * dofs_per_entity[d] + effective_rank
        n_cells = topology.n_cells
        cell_dofs = torch.empty(n_cells, len(layout), dtype=torch.int64)
        for i, dof in enumerate(layout):
            ed, lei = dof.entity_dim, dof.local_index
            dpe = self._dofs_per_entity[ed]
            global_entity = topology.cell_entities(ed)[:, lei]  # (n_cells,)
            # Reversal (A1 orientation): on an entity the cell traverses backwards
            # (``entity_orientation == -1``) its DOF ranks run in reverse, so both cells sharing
            # the entity agree on which global DOF each point owns. A no-op for an entity with a
            # single DOF (``dpe == 1``) or no orientation (vertices/cells, sign ``+1``) -- hence
            # P1/P2/DG0/Hermite numbering is unchanged; it only acts on >=2 DOFs per reversed edge.
            sign = topology.entity_orientation(ed)[:, lei]  # (n_cells,) +/-1
            eff_rank = torch.where(
                sign == 1,
                torch.full_like(sign, rank[i]),
                torch.full_like(sign, dpe - 1 - rank[i]),
            )  # (n_cells,)
            cell_dofs[:, i] = self._offset[ed] + global_entity * dpe + eff_rank

        # Periodic identification + compaction: remap[old_global] -> new_global.
        remap = self._build_remap(total, periodic)
        self.register_buffer("_remap", remap)
        self.n_scalar_dofs = int(remap.max().item()) + 1 if total else 0
        self.register_buffer("cell_dofs", remap[cell_dofs])

    def _build_remap(self, total, periodic) -> torch.Tensor:
        """Union periodic vertex-DOF pairs, then compact to contiguous ids.

        Returns ``remap`` of shape ``(total,)`` mapping each naive global DOF to its
        (possibly merged) compacted id. Identity when ``periodic`` is empty -- the common case,
        returned without touching every DOF (the Python union-find below ran over all ``total`` DOFs
        and dominated the DOF-map build).
        """
        if not periodic:
            return torch.arange(total, dtype=torch.int64)

        parent = list(range(total))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        dpe0, off0 = self._dofs_per_entity.get(0, 0), self._offset.get(0, 0)
        for a, b in periodic:
            for r in range(dpe0):  # match DOFs by rank (value<->value, d1<->d1, ...)
                ra, rb = find(off0 + a * dpe0 + r), find(off0 + b * dpe0 + r)
                parent[max(ra, rb)] = min(ra, rb)

        # Resolve every DOF to its group root (vectorised pointer-jumping), then compact to
        # contiguous ids. Each root is the MIN of its group (``parent[max]=min`` above), so it first
        # appears at its own index -- sorted order == first-appearance order, and ``torch.unique``'s
        # inverse reproduces the original first-appearance numbering exactly.
        parent_t = torch.tensor(parent, dtype=torch.int64)
        roots = torch.arange(total, dtype=torch.int64)
        while True:
            nxt = parent_t[roots]
            if torch.equal(nxt, roots):
                break
            roots = nxt
        _, remap = torch.unique(roots, return_inverse=True)
        return remap.to(torch.int64)

    @property
    def n_local_dofs(self) -> int:
        """Number of local (per-cell) DOFs = ``len(element.dof_layout)``."""
        return self.cell_dofs.shape[1]

    def locate(self, entity_dim: int | EntityDim, entities, kinds=None) -> torch.Tensor:
        """Global scalar-DOF indices for given entities and DOF kinds.

        Args:
            entity_dim (int | EntityDim): Dimension of the entities
                (``EntityDim.VERTEX`` vertices, ``topology.cell_dim`` cells).
            entities (Sequence[int]): Global entity indices.
            kinds (Sequence[str], optional): DOF kinds to select on each entity;
                defaults to all kinds carried by that entity dimension.

        Returns:
            torch.Tensor: Sorted global scalar-DOF indices.
        """
        all_kinds = self._kinds_per_entity[entity_dim]
        if kinds is None:
            ranks = torch.arange(len(all_kinds), dtype=torch.int64)
        else:
            # Every DOF whose kind is requested -- so a repeated kind (e.g. two P3 edge values)
            # returns both its ranks, not just the first.
            ranks = torch.tensor(
                [r for r, k in enumerate(all_kinds) if k in kinds], dtype=torch.int64
            )
        dpe, off = self._dofs_per_entity[entity_dim], self._offset[entity_dim]
        entities = torch.as_tensor(list(entities), dtype=torch.int64)
        # Naive global index of every (entity, rank) pair, as an outer sum; route through the
        # periodic/compaction remap, then dedup+sort (torch.unique). Empty entities/ranks -> empty.
        naive = (off + entities.reshape(-1, 1) * dpe + ranks.reshape(1, -1)).reshape(-1)
        return torch.unique(self._remap[naive])
