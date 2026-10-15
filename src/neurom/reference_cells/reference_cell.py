"""Reference cells (plain, immutable, dtype/device-agnostic) and the library instances."""

import torch

from neurom.meshes.topology import EntityDim


class ReferenceCell:
    """Reference simplex geometry: vertices, measure, and local sub-entities.

    A plain immutable value object: a single shared instance can serve
    every mesh, dtype and device because no tensor is pinned to a fixed dtype. ``simplex`` and
    ``measure`` are built on access in ``torch.get_default_dtype()`` from a float64 master, so a
    float64 run gets float64 reference coordinates regardless of when the cell was created -- and
    casting one model never mutates another's shared cell.

    Concrete cells are module-level instances (:data:`BAR`, :data:`TRIANGLE`), not subclasses --
    a cell carries only data (vertices, measure, sub-entities), no behaviour.

    Local entities are exposed by :class:`~neurom.meshes.EntityDim` via :meth:`local_entities`
    (``VERTEX``/``EDGE``/``FACE``/``VOLUME``) -- the single source of the local-entity convention,
    shared with :class:`~neurom.meshes.Topology`.

    Attributes:
        simplex (torch.Tensor): Vertex coordinates ``(n_nodes, dim_ref)`` in the active dtype.
        measure (torch.Tensor): Measure (length/area/volume) as a scalar in the active dtype.
    """

    def __init__(self, simplex, measure, edges=None):
        """Store the reference geometry as dtype-agnostic masters.

        Args:
            simplex: Vertex coordinates ``(n_nodes, dim_ref)`` (tensor or nested list).
            measure: Scalar measure (length/area/volume) of the reference cell.
            edges: ``(n_edges, 2)`` local vertex-index pairs of the cell's proper sub-edges
                (dimension 1), e.g. a triangle's ``[[0,1],[1,2],[2,0]]``. Omitted when the cell has
                no proper sub-edges (a bar, whose only 1-D entity is the cell itself).

        Raises:
            ValueError: If ``simplex`` is not 2-D, or ``edges`` is given for a cell that has no
                proper sub-edges.
        """
        simplex = torch.as_tensor(simplex, dtype=torch.float64)
        if simplex.ndim != 2:
            raise ValueError("simplex must be (n_nodes, dim_ref)")
        self._simplex = simplex  # float64 master; cast to the active dtype on access
        self._measure = float(measure)
        # Proper sub-entities declared by the caller, keyed by absolute topological dimension.
        # Vertices (dim 0) and the top-dimensional cell (dim = dim_ref) are generated on the fly.
        # A declared sub-entity must be *proper* (0 < dim < dim_ref) so it can never collide with
        # the vertex or cell branch of local_entities -- e.g. a bar (dim_ref 1) has no sub-edges,
        # its only 1-D entity being the cell itself.
        dim_ref = simplex.shape[1]
        self._sub_entities: dict[int, torch.Tensor] = {}
        if edges is not None:
            if dim_ref <= 1:
                raise ValueError(
                    f"edges are a proper sub-entity (dimension 1); a cell of dimension "
                    f"{dim_ref} has none (its only 1-D entity is the cell itself)"
                )
            self._sub_entities[1] = torch.as_tensor(edges, dtype=torch.int64)

    @property
    def simplex(self) -> torch.Tensor:
        """Vertex coordinates ``(n_nodes, dim_ref)`` in the active default dtype."""
        return self._simplex.to(torch.get_default_dtype())

    @property
    def measure(self) -> torch.Tensor:
        """Reference measure (scalar) in the active default dtype."""
        return torch.tensor(self._measure, dtype=torch.get_default_dtype())

    @property
    def dim_ref(self) -> int:
        """Reference (parametric) dimension: 1 for a bar, 2 for a triangle."""
        return self._simplex.shape[1]

    def local_entities(self, entity_dim: EntityDim) -> torch.Tensor:
        """Local vertex-index tuples of the cell's entities of a given :class:`EntityDim`.

        Keyed by **absolute** topological dimension, independent of the mesh:
        :attr:`~neurom.meshes.EntityDim.VERTEX`, ``EDGE``, ``FACE``, ``VOLUME`` -- so edges are
        always ``local_entities(EntityDim.EDGE)`` in 2-D and 3-D alike. Vertices and the
        top-dimensional cell are generated here; proper sub-entities (a triangle's edges) are
        declared at construction. Examples -- triangle: ``VERTEX -> [[0],[1],[2]]``,
        ``EDGE -> [[0,1],[1,2],[2,0]]``, ``FACE -> [[0,1,2]]``; bar: ``VERTEX -> [[0],[1]]``,
        ``EDGE -> [[0,1]]`` (the cell is the edge).

        Args:
            entity_dim (EntityDim): Absolute topological dimension of the entities (a plain
                ``int`` is accepted too).

        Returns:
            torch.Tensor: ``(n_entities, n_verts_per_entity)`` local vertex indices (int64).

        Raises:
            ValueError: If the cell has no entities of that dimension.
        """
        d = int(entity_dim)
        n_verts = self._simplex.shape[0]
        if d == EntityDim.VERTEX:  # dimension 0
            return torch.arange(n_verts, dtype=torch.int64).unsqueeze(1)  # (n, 1)
        if d == self.dim_ref:  # the top-dimensional cell itself
            return torch.arange(n_verts, dtype=torch.int64).unsqueeze(0)  # (1, n)
        if d in self._sub_entities:  # declared sub-entities (e.g. a triangle's edges)
            return self._sub_entities[d]
        raise ValueError(
            f"reference cell of dimension {self.dim_ref} has no entities of dimension {d}"
        )


BAR = ReferenceCell(simplex=[[-1.0], [1.0]], measure=2.0)
r"""**Reference bar** on :math:`[-1, 1]`: two vertices, length 2, no proper sub-edges (its only
1-D entity is the cell itself)."""

TRIANGLE = ReferenceCell(
    simplex=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    measure=0.5,
    edges=[[0, 1], [1, 2], [2, 0]],  # cyclic (0,1),(1,2),(2,0)
)
r"""**Reference triangle** with vertices ``(0,0),(1,0),(0,1)``, area :math:`1/2`. Edges follow the
cyclic convention ``(0,1),(1,2),(2,0)`` (``local_entities(EntityDim.EDGE)``) -- the single source
shared with :class:`~neurom.meshes.Topology`."""
