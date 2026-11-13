"""WP9 stage B/C: EntityGeometry -- per-facet orientation signs, normals and measures.

Pure geometry/topology of the mesh facets (edges in 2-D): the orientation sign sigma_{e,s} that two
cells sharing a facet must agree on, and the canonical outward normal n_E and measure |E| a flux
(H(div)) element consumes. The *use* of sigma as the T_e factor inside a Transformation lives with
the Transformation tests, not here.
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import EntityGeometry
from neurom.elements import P1_TRIANGLE, VectorElement, EntityDim

F64 = torch.float64
INV_SQRT2 = 0.5 ** 0.5


def _square():
    """Unit square as two triangles sharing the diagonal edge (0, 2): (topology, mesh)."""
    topo = Topology(torch.tensor([[0, 1, 2], [0, 2, 3]]))
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=F64)
    mesh = Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))
    return topo, mesh


def test_facet_signs_are_the_edge_orientation():
    """sigma_{e,s} matches hand-derived +/-1 from the cyclic-edge convention (and Topology agrees)."""
    topo, mesh = _square()
    eg = EntityGeometry(mesh)
    assert eg.facet_dim == EntityDim.EDGE  # 2-D: facets are edges

    # Ground truth, hand-derived from the convention (not from the code under test): cyclic local
    # edges (0,1),(1,2),(2,0); sigma = +1 iff the local vertex order matches canonical (sorted).
    #   cell [0,1,2]: e0=(0,1)=canon +1, e1=(1,2)=canon +1, e2=(2,0) vs canon (0,2) -> -1
    #   cell [0,2,3]: e0=(0,2)=canon +1, e1=(2,3)=canon +1, e2=(3,0) vs canon (0,3) -> -1
    expected = torch.tensor([[1, 1, -1], [1, 1, -1]])
    assert torch.equal(eg.facet_signs(), expected)
    # and it is exactly what Topology reports (delegation), on the full set and a subset
    assert torch.equal(eg.facet_signs(), topo.entity_orientation(EntityDim.EDGE))
    ids = torch.tensor([1])
    assert torch.equal(eg.facet_signs(ids), expected[ids])


def test_facet_normals_and_measures_match_hand_computed_values():
    """n_E equals the hand-computed outward n = R t_hat (fixes the R sign) and |E| the edge length.

    The generic unit/perpendicular invariants hold for BOTH R = +-90 deg, so they can't pin the
    convention down; the explicit normals (which distinguish outward R = -90 from inward) do.

    Global edges (canonical lo,hi): e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(2,3); n=R t_hat gives
    e0->(0,-1)|1, e1->(1,-1)/r2|r2, e2->(1,0)|1, e3->(1,0)|1, e4->(0,1)|1.  cell_entities maps
      cell 0 = [0,1,2] -> edges [0,3,1] -> [(0,-1),  (1,0), (1,-1)/r2],  |E| [1, 1, r2]
      cell 1 = [0,2,3] -> edges [1,4,2] -> [(1,-1)/r2, (0,1), (1,0)  ],  |E| [r2, 1, 1]
    (edge 1, the shared diagonal, hands the SAME (1,-1)/r2 to both cells -- the per-cell flip is sigma.)
    """
    topo, mesh = _square()
    eg = EntityGeometry(mesh)
    n = eg.facet_normals()  # (2, 3, 2)
    L = eg.facet_measures()  # (2, 3, 1)
    assert n.shape == (2, 3, 2) and L.shape == (2, 3, 1)

    r2 = 2.0 ** 0.5
    expected_n = torch.tensor(
        [[[0.0, -1.0], [1.0, 0.0], [INV_SQRT2, -INV_SQRT2]],  # cell 0
         [[INV_SQRT2, -INV_SQRT2], [0.0, 1.0], [1.0, 0.0]]],  # cell 1
        dtype=F64,
    )
    expected_L = torch.tensor([[[1.0], [1.0], [r2]], [[r2], [1.0], [1.0]]], dtype=F64)
    assert n.numpy() == pytest.approx(expected_n.numpy(), abs=1e-12)  # has 0 components -> abs
    assert L.numpy() == pytest.approx(expected_L.numpy(), rel=1e-12)

    # Invariants that must hold for every facet: unit length, and perpendicular to the edge tangent.
    assert n.norm(dim=-1).numpy() == pytest.approx(torch.ones(2, 3, dtype=F64).numpy(), rel=1e-12)
    ce = topo.cell_entities(EntityDim.EDGE)  # (2, 3) local -> global edge
    ev, x = topo.edge_vertices, mesh.coordinates.full_values()
    for e in range(2):
        for s in range(3):
            t = x[ev[ce[e, s], 1]] - x[ev[ce[e, s], 0]]  # edge tangent lo -> hi
            assert float(n[e, s] @ t) == pytest.approx(0.0, abs=1e-12)  # normal _|_ tangent
            assert float(L[e, s, 0]) == pytest.approx(float(t.norm()), rel=1e-12)
