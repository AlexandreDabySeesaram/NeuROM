import torch

from neurom.meshes.topology import Topology
from neurom.elements import (
    P1_BAR,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
    P2_TRIANGLE,
    EntityDim,
)
from neurom.function_space.dof_map import DofMap

F64 = torch.float64


def _bar_topology(N=5):
    # N vertices, N-1 cells on a line
    cell_vertices = torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T
    return Topology(cell_vertices), N


def test_topology_basics():
    topo, N = _bar_topology(5)
    assert topo.dim == 1
    assert topo.n_vertices == N
    assert topo.n_cells == N - 1
    assert torch.equal(topo.cell_entities(0), topo.cell_vertices)
    assert torch.equal(topo.cell_entities(1), torch.arange(N - 1).unsqueeze(-1))


def test_entity_dim_enum_and_relative_helpers():
    from neurom.meshes import EntityDim

    topo, _ = _bar_topology(5)
    # absolute dims are plain ints
    assert EntityDim.VERTEX == 0 and EntityDim.EDGE == 1
    # relative helpers: cell = top dim (1 here), facet = dim - 1
    assert topo.cell_dim == 1 and topo.facet_dim == 0
    # enum members work anywhere an entity-dim int is expected
    assert torch.equal(topo.cell_entities(EntityDim.VERTEX), topo.cell_vertices)
    assert topo.n_entities(EntityDim.VERTEX) == topo.n_vertices


def test_element_deduces_dof_nodes_and_counts():
    hermite = HERMITE
    assert hermite.n_dofs == 4
    assert hermite.dof_kinds == ("value", "d1", "value", "d1")
    # deduced from dof_layout + Bar vertices [-1, 1]
    assert torch.equal(
        hermite.dof_nodes_xi().flatten(), torch.tensor([-1.0, -1.0, 1.0, 1.0])
    )
    assert torch.equal(P1_BAR.dof_nodes_xi().flatten(), torch.tensor([-1.0, 1.0]))


def test_dof_layout_entity_dim_is_entity_dim():
    # entity_dim in the layout is a typed EntityDim (not a bare int).
    assert HERMITE.dof_layout[0][0] is EntityDim.VERTEX
    assert all(isinstance(e, EntityDim) for e, _, _ in HERMITE.dof_layout)
    # a cell DOF carries the cell's *absolute* dimension: bar cell = EDGE, triangle = FACE.
    assert DG0_BAR.dof_layout[0][0] == EntityDim.EDGE
    assert DG0_TRIANGLE.dof_layout[0][0] == EntityDim.FACE
    assert isinstance(DG0_TRIANGLE.dof_layout[0][0], EntityDim)


def test_p1_dofmap_equals_vertex_connectivity():
    topo, N = _bar_topology(5)
    dm = DofMap(topo, P1_BAR)
    assert dm.n_scalar_dofs == N
    assert torch.equal(dm.cell_dofs, topo.cell_vertices)  # one DOF per vertex


def test_hermite_dofmap_two_dofs_per_vertex():
    topo, N = _bar_topology(5)
    dm = DofMap(topo, HERMITE)
    assert dm.n_scalar_dofs == 2 * N
    # cell e gathers [2e, 2e+1, 2e+2, 2e+3]
    expected = torch.stack([torch.arange(2 * e, 2 * e + 4) for e in range(N - 1)])
    assert torch.equal(dm.cell_dofs, expected)


def test_dg0_dofmap_one_dof_per_cell():
    topo, N = _bar_topology(5)
    dm = DofMap(topo, DG0_BAR)
    assert dm.n_scalar_dofs == N - 1
    assert torch.equal(dm.cell_dofs, torch.arange(N - 1).unsqueeze(-1))


def test_locate_hermite_clamped_ends():
    topo, N = _bar_topology(5)
    dm = DofMap(topo, HERMITE)
    dofs = dm.locate(0, entities=[0, N - 1], kinds=["value", "d1"])
    assert torch.equal(dofs, torch.tensor([0, 1, 2 * N - 2, 2 * N - 1]))
    # value only (pinned, slope free)
    pinned = dm.locate(0, entities=[0], kinds=["value"])
    assert torch.equal(pinned, torch.tensor([0]))


def test_periodic_p1_identifies_end_vertices():
    # Identify the last vertex with the first: one fewer DOF, both ends share it.
    topo, N = _bar_topology(5)
    dm = DofMap(topo, P1_BAR, periodic=[(0, N - 1)])
    assert dm.n_scalar_dofs == N - 1  # one vertex collapsed
    # the first and last cell now reference the SAME global DOF at the seam
    assert dm.cell_dofs[0, 0].item() == dm.cell_dofs[-1, 1].item()
    # locate on either identified vertex returns the one shared DOF
    assert torch.equal(dm.locate(0, [0]), dm.locate(0, [N - 1]))


def test_periodic_hermite_identifies_value_and_slope():
    # Both DOFs (value and slope) of the seam vertices are matched by rank.
    topo, N = _bar_topology(5)
    dm = DofMap(topo, HERMITE, periodic=[(0, N - 1)])
    assert dm.n_scalar_dofs == 2 * N - 2  # 2 DOFs collapsed
    first = dm.locate(0, [0], kinds=["value", "d1"])
    last = dm.locate(0, [N - 1], kinds=["value", "d1"])
    assert torch.equal(first, last)


def test_periodic_shares_one_trainable_parameter():
    # End-to-end: a field on a periodic space has u(0) == u(N-1) by construction,
    # with gradient flowing to a single shared parameter.
    from neurom.meshes import Mesh
    from neurom.fields import Field, TrainableField
    from neurom.function_space import FunctionSpace
    from neurom.elements import VectorElement

    N = 5
    cell_vertices = torch.vstack([torch.arange(N - 1), torch.arange(1, N)]).T
    points = torch.linspace(0, 1, N).unsqueeze(-1)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, points))
    space = FunctionSpace(mesh.topology, P1_BAR, periodic=[(0, N - 1)])

    init = torch.arange(space.n_scalar_dofs, dtype=F64).unsqueeze(-1)
    u = TrainableField(space, init, name="u")
    # at_elements gathers per-cell DOFs: the seam ends (cell 0 left, last cell right)
    # resolve to the SAME global DOF, so they carry one identical value.
    per_cell = u.at_elements().squeeze(-1)  # (n_cells, n_local)
    assert per_cell[0, 0].item() == per_cell[-1, 1].item()  # u(0) == u(N-1)
    assert u.values_reduced.numel() == N - 1  # one shared parameter at the seam


def _two_triangle_square():
    # Unit square split into two triangles sharing the diagonal edge (0, 2).
    return Topology(torch.tensor([[0, 1, 2], [0, 2, 3]]))


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


def test_p2_dofmap_numbers_vertices_then_edges():
    topo = _two_triangle_square()
    dm = DofMap(topo, P2_TRIANGLE)
    # 4 vertices + 5 edges, one value DOF each
    assert dm.n_scalar_dofs == 4 + 5
    # each cell gathers its 3 vertex DOFs then its 3 edge DOFs (edges offset past vertices)
    assert dm.cell_dofs.shape == (2, 6)
    assert torch.equal(dm.cell_dofs[:, :3], topo.cell_vertices)
    assert torch.equal(dm.cell_dofs[:, 3:], 4 + topo.cell_entities(EntityDim.EDGE))
    # the shared diagonal edge -> one shared global DOF (cell 0 local edge 2, cell 1 local 0)
    assert dm.cell_dofs[0, 5].item() == dm.cell_dofs[1, 3].item()


def test_p2_dof_nodes_xi_places_edge_dofs_at_reference_midpoints():
    xi = P2_TRIANGLE.dof_nodes_xi()  # (6, 2)
    # vertices then edge midpoints of the reference triangle (0,0),(1,0),(0,1)
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    )
    assert torch.allclose(xi, expected)
