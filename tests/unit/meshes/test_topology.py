"""Topology: entity counts, cell-to-entity incidence, edge enumeration/orientation, boundary."""

import torch

from neurom.meshes import Topology, EntityDim


def _bar(N=5):
    """N vertices, N-1 cells on a line."""
    return Topology(torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T)


def _two_triangle_square():
    """Unit square split into two triangles sharing the diagonal edge (0, 2)."""
    return Topology(torch.tensor([[0, 1, 2], [0, 2, 3]]))


def _square(n):
    """Structured n x n unit-square triangle mesh (alternating diagonals): topology only."""
    vid = lambda i, j: i * (n + 1) + j
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return Topology(torch.tensor(cells))


# --- entities & incidence -----------------------------------------------------


def test_topology_basics():
    topo, N = _bar(5), 5
    assert topo.dim == 1
    assert topo.n_vertices == N
    assert topo.n_cells == N - 1
    assert torch.equal(topo.cell_entities(0), topo.cell_vertices)
    assert torch.equal(topo.cell_entities(1), torch.arange(N - 1).unsqueeze(-1))


def test_entity_dim_enum_and_relative_helpers():
    topo = _bar(5)
    # absolute dims are plain ints
    assert EntityDim.VERTEX == 0 and EntityDim.EDGE == 1
    # relative helpers: cell = top dim (1 here), facet = dim - 1
    assert topo.cell_dim == 1 and topo.facet_dim == 0
    # enum members work anywhere an entity-dim int is expected
    assert torch.equal(topo.cell_entities(EntityDim.VERTEX), topo.cell_vertices)
    assert topo.n_entities(EntityDim.VERTEX) == topo.n_vertices


# --- edges --------------------------------------------------------------------


def test_edge_enumeration_shares_the_interior_edge():
    topo = _two_triangle_square()
    assert topo.n_entities(EntityDim.EDGE) == 5  # 4 boundary + 1 diagonal
    ce = topo.cell_entities(EntityDim.EDGE)  # (2, 3)
    assert ce.shape == (2, 3)
    shared = set(ce[0].tolist()) & set(ce[1].tolist())
    assert len(shared) == 1  # the diagonal is one global edge shared by both cells
    (edge_id,) = shared
    assert sorted(topo.edge_vertices[edge_id].tolist()) == [0, 2]  # the diagonal (0, 2)


def test_edge_orientation_is_signed():
    topo = _two_triangle_square()
    orient = topo.entity_orientation(EntityDim.EDGE)
    assert orient.shape == (2, 3)
    assert set(orient.flatten().tolist()) <= {-1, 1}
    # local edge (2, 0) is stored reversed vs canonical (0, 2) -> -1 in both cells
    assert orient[0, 2] == -1 and orient[1, 2] == -1


def test_edges_are_built_lazily():
    """(T-WP6.3) edges are built on first access, so a P1-only problem never pays for them."""
    topo = _two_triangle_square()
    assert topo._edges_built is False  # nothing built at construction
    # vertex / cell access must not trigger the edge build
    _ = topo.cell_vertices
    _ = topo.n_cells
    _ = topo.cell_entities(EntityDim.VERTEX)
    _ = topo.n_entities(EntityDim.VERTEX)
    assert topo._edges_built is False
    # the first edge access builds them (and only then)
    _ = topo.n_entities(EntityDim.EDGE)
    assert topo._edges_built is True


# --- boundary (WP7) -----------------------------------------------------------


def test_boundary_counts_on_the_unit_square():
    """(T-WP7.1) an n x n square has 4n boundary edges and 4n boundary vertices."""
    for n in (2, 3, 4):
        topo = _square(n)
        assert topo.boundary(EntityDim.EDGE).numel() == 4 * n
        assert topo.boundary(EntityDim.VERTEX).numel() == 4 * n


def test_boundary_vertices_lie_on_boundary_edges():
    """The boundary vertices are exactly the vertices of the boundary edges."""
    topo = _square(3)
    bedges = topo.boundary(EntityDim.EDGE)
    from_edges = torch.unique(topo.edge_vertices[bedges].reshape(-1))
    assert torch.equal(topo.boundary(EntityDim.VERTEX), from_edges)


def test_boundary_of_a_bar_is_the_two_ends():
    """(T-WP7.1, 1-D) the boundary vertices of a bar are its two end points (facets = vertices)."""
    topo = _bar(5)  # vertices 0..4
    assert sorted(topo.boundary(EntityDim.VERTEX).tolist()) == [0, 4]


def test_boundary_of_an_unsupported_dimension_raises():
    import pytest

    topo = _square(2)
    with pytest.raises(NotImplementedError):
        topo.boundary(topo.cell_dim)  # cells are not boundary entities
