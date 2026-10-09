"""Pure-integer mesh topology: entities per dimension and cell-to-entity incidence.

Built from the cell-to-vertex table alone (no positions). Provides, for each entity
dimension :math:`d`, the number of entities, the cell-to-entity incidence table, and the
local-vs-global orientation. Vertices (:math:`d = 0`) and cells (:math:`d = \\dim`) are
always available; **edges** (:math:`d = 1`, in 2-D+) are enumerated by sort+unique over the
cells' local edges -- the shared-DOF layer P2/Morley/RT need. Faces (:math:`d = 2` in 3-D)
are the remaining hook, not built yet.
"""

from enum import IntEnum

import torch
import torch.nn as nn


class EntityDim(IntEnum):
    """Topological entity dimension, as a named :class:`int`.

    These are the *absolute* dimensions. "Cell" and "facet" are *relative* to the mesh
    (a cell is a triangle in 2-D but a bar in 1-D), so they are not enum members; use
    :attr:`Topology.cell_dim` and :attr:`Topology.facet_dim` for those. Because the enum
    subclasses :class:`int`, a member may be used anywhere an entity-dimension ``int`` is
    expected.

    Attributes:
        VERTEX (int): Point, dimension ``0``.
        EDGE (int): Line, dimension ``1``.
        FACE (int): Surface, dimension ``2``.
        VOLUME (int): Cell volume, dimension ``3``.
    """

    VERTEX = 0
    EDGE = 1
    FACE = 2
    VOLUME = 3


class Topology(nn.Module):
    """Simplicial mesh topology derived from a cell-to-vertex table.

    Args:
        cell_vertices (torch.Tensor): Integer tensor of shape
            ``(n_cells, n_verts_per_cell)``. ``n_verts_per_cell`` is ``2`` for a
            1-D bar mesh and ``3`` for a 2-D triangle mesh, so the topological
            dimension is ``n_verts_per_cell - 1``.
        n_vertices (int, optional): Total number of vertices. Defaults to
            ``cell_vertices.max() + 1``.

    Attributes:
        cell_vertices (torch.Tensor): The cell-to-vertex table (registered buffer).
    """

    def __init__(self, cell_vertices: torch.Tensor, n_vertices: int | None = None):
        super().__init__()
        if cell_vertices.ndim != 2:
            raise ValueError("cell_vertices must be (n_cells, n_verts_per_cell)")
        self.register_buffer("cell_vertices", cell_vertices)
        self._dim = cell_vertices.shape[1] - 1
        self._n_vertices = (
            int(cell_vertices.max()) + 1 if n_vertices is None else n_vertices
        )
        # Edges are intermediate entities, built only in 2-D+ (in 1-D an "edge" is the cell).
        self._n_edges = 0
        if self._dim >= 2:
            self._build_edges()

    @staticmethod
    def _simplex_local_edges(n_verts_per_cell: int) -> torch.Tensor:
        """Local vertex-index pairs of a simplex cell's edges (shared with reference cells)."""
        if n_verts_per_cell == 3:  # triangle: cyclic (0,1),(1,2),(2,0)
            return torch.tensor([[0, 1], [1, 2], [2, 0]])
        raise NotImplementedError(
            f"edge enumeration is only built for triangles; got "
            f"{n_verts_per_cell}-vertex cells."
        )

    def _build_edges(self):
        """Enumerate global edges by sort+unique over the cells' local edges (cached)."""
        local = self._simplex_local_edges(self.cell_vertices.shape[1])  # (n_local, 2)
        pairs = self.cell_vertices[
            :, local
        ]  # (n_cells, n_local, 2) global vertex pairs
        flat = pairs.reshape(-1, 2)
        canon, _ = torch.sort(flat, dim=1)  # (min, max) so shared edges match
        edge_vertices, inverse = torch.unique(canon, dim=0, return_inverse=True)
        n_local = local.shape[0]
        self._n_edges = edge_vertices.shape[0]
        self.register_buffer("edge_vertices", edge_vertices)  # (n_edges, 2)
        self.register_buffer(
            "_cell_edges", inverse.reshape(-1, n_local)
        )  # (n_cells, n_local)
        # +1 where the cell's local edge is already in canonical order, else -1 (the sign the
        # two cells sharing an edge must agree on for orientation-sensitive DOFs).
        signs = torch.where((flat == canon).all(dim=1), 1, -1)
        self.register_buffer("_edge_orientation", signs.reshape(-1, n_local))

    @property
    def dim(self) -> int:
        """Topological dimension (1 for bars, 2 for triangles)."""
        return self._dim

    @property
    def cell_dim(self) -> int:
        """Dimension of a cell -- the top dimension, equal to :attr:`dim`."""
        return self._dim

    @property
    def facet_dim(self) -> int:
        """Dimension of a facet -- one below a cell (:attr:`dim` ``- 1``)."""
        return self._dim - 1

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.cell_vertices.shape[0]

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return self._n_vertices

    def n_entities(self, d: int) -> int:
        """Number of entities of dimension ``d``.

        Args:
            d (int | EntityDim): Entity dimension (``0`` vertices, :attr:`dim` cells).

        Returns:
            int: The entity count.

        Raises:
            NotImplementedError: For intermediate entities (edges/faces), not built yet.
        """
        if d == EntityDim.VERTEX:
            return self.n_vertices
        if d == EntityDim.EDGE and self._dim >= 2:
            return self._n_edges
        if d == self.cell_dim:
            return self.n_cells
        raise NotImplementedError(
            f"Entities of dimension {int(d)} are not built yet; only vertices (0), "
            "edges (1, in 2-D+) and cells (dim) are available in this cut."
        )

    def cell_entities(self, d: int) -> torch.Tensor:
        """Cell-to-entity incidence for entity dimension ``d``.

        Args:
            d (int | EntityDim): Entity dimension.

        Returns:
            torch.Tensor: ``(n_cells, n_local_entities)`` global entity indices.
            For ``d == 0`` this is ``cell_vertices``; for ``d == dim`` it is the
            identity ``arange(n_cells)[:, None]`` (each cell is its own entity).

        Raises:
            NotImplementedError: For intermediate entities (edges/faces), not built yet.
        """
        if d == EntityDim.VERTEX:
            return self.cell_vertices
        if d == EntityDim.EDGE and self._dim >= 2:
            return self._cell_edges
        if d == self.cell_dim:
            return torch.arange(
                self.n_cells, device=self.cell_vertices.device
            ).unsqueeze(-1)
        raise NotImplementedError(f"cell_entities({int(d)}) is not built yet.")

    def entity_orientation(self, d: int) -> torch.Tensor:
        """Local-vs-global orientation sign of each cell's entities.

        Args:
            d (int | EntityDim): Entity dimension.

        Returns:
            torch.Tensor: ``(n_cells, n_local_entities)`` of ``±1``. Vertices and cells carry
            no orientation (all ``+1``); an edge is ``+1`` where the cell's local edge is in
            canonical (sorted-vertex) order and ``-1`` otherwise -- the sign the two cells
            sharing an edge must agree on for orientation-sensitive DOFs (Morley/RT).
        """
        if d == EntityDim.EDGE and self._dim >= 2:
            return self._edge_orientation
        return torch.ones_like(self.cell_entities(d), dtype=torch.int64)
