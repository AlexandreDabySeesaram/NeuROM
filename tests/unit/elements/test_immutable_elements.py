"""Elements, reference cells and shape functions are plain, immutable and dtype-agnostic, and the
reference cell is the single source of the local-entity convention.

These guard three properties: no shared mutable state -- a model casting itself cannot corrupt a
singleton element shared with another model; a float64 run uses float64 reference coordinates end
to end; and Topology, the reference Triangle and the P2 basis agree on the local-edge convention,
which is defined once.
"""

import pytest
import torch
import torch.nn as nn

from neurom.elements import P1_BAR, P1_TRIANGLE, P2_TRIANGLE, HERMITE, EntityDim
from neurom.reference_cells import BAR, TRIANGLE
from neurom.meshes.topology import Topology

ELEMENTS = [P1_BAR, P1_TRIANGLE, P2_TRIANGLE, HERMITE]
IDS = ["P1_BAR", "P1_TRIANGLE", "P2_TRIANGLE", "HERMITE"]


@pytest.fixture(autouse=True)
def _restore_dtype():
    """Each test sets the default dtype explicitly; restore it afterwards."""
    prev = torch.get_default_dtype()
    yield
    torch.set_default_dtype(prev)


# ------------------------------------------------------------------- no shared mutable state


@pytest.mark.parametrize("element", ELEMENTS, ids=IDS)
def test_elements_are_plain_objects_not_modules(element):
    """The element and the reference cell / basis it holds are plain objects (not ``nn.Module``).

    This is what makes the no-shared-state guarantee hold structurally: a model's
    ``.to(dtype/device)`` walks its module tree, and because the element is not in it, no shared
    singleton tensor can be cast out from under another model.
    """
    assert not isinstance(element, nn.Module)
    assert not isinstance(element.cell, nn.Module)
    assert not isinstance(element.reference_basis, nn.Module)


def test_shared_element_serves_both_dtypes_without_pinned_state():
    """The same singleton yields float32 in a float32 context and float64 in a float64 context.

    No reference tensor is pinned to a dtype, so there is nothing one model could cast and thereby
    corrupt for another sharing the element -- the reference coordinates are rebuilt on access.
    """
    torch.set_default_dtype(torch.float32)
    s32 = P1_TRIANGLE.cell.simplex
    torch.set_default_dtype(torch.float64)
    s64 = P1_TRIANGLE.cell.simplex
    assert s32.dtype == torch.float32
    assert s64.dtype == torch.float64
    assert torch.equal(s32.double(), s64)  # same geometry, only the dtype differs


# ----------------------------------------------------------------------- float64 end to end


@pytest.mark.parametrize("element", ELEMENTS, ids=IDS)
def test_float64_reference_coordinates_end_to_end(element):
    """In a float64 context every reference tensor the element exposes is float64.

    Covers the cell geometry (``simplex``, ``measure``), the deduced DOF coordinates
    (``dof_nodes_xi``) and the basis evaluation (``N``) -- the paths that previously leaked a
    float32 reference built at import time.
    """
    torch.set_default_dtype(torch.float64)
    assert element.cell.simplex.dtype == torch.float64
    assert element.cell.measure.dtype == torch.float64
    assert element.dof_nodes_xi().dtype == torch.float64
    assert element.N(element.cell.simplex[:1].unsqueeze(0)).dtype == torch.float64


# --------------------------------------------------------------------- one local-edge source


def test_single_local_edge_convention():
    """Topology, the reference Triangle and the P2 basis agree on the local-edge convention.

    The convention ``(0,1),(1,2),(2,0)`` is defined once -- on the reference cell -- and Topology
    reads it from there rather than redefining it.
    """
    tri = TRIANGLE.local_entities(EntityDim.EDGE)
    assert torch.equal(tri, torch.tensor([[0, 1], [1, 2], [2, 0]]))
    # Topology borrows the convention from the reference cell (single source).
    assert torch.equal(Topology._simplex_local_edges(3), tri)
    # The P2 edge DOFs are ordered e01, e12, e20 -- the same edge order the basis columns follow.
    edge_dofs = [d for d in P2_TRIANGLE.dof_layout if d.entity_dim == EntityDim.EDGE]
    assert [d.local_index for d in edge_dofs] == [0, 1, 2]


def test_local_entities_by_absolute_dimension():
    """``local_entities(dim)`` is keyed by absolute topological dimension, mesh-independent.

    0 = vertices, 1 = edges, 2 = faces -- edges are ``local_entities(1)`` in every dimension.
    Vertices and the top-dimensional cell are generated; proper sub-entities are declared.
    """
    tri = TRIANGLE
    assert torch.equal(tri.local_entities(0), torch.tensor([[0], [1], [2]]))  # vertices
    assert torch.equal(
        tri.local_entities(1), torch.tensor([[0, 1], [1, 2], [2, 0]])
    )  # edges
    assert torch.equal(
        tri.local_entities(2), torch.tensor([[0, 1, 2]])
    )  # the cell/face

    bar = BAR
    assert torch.equal(bar.local_entities(0), torch.tensor([[0], [1]]))  # vertices
    assert torch.equal(
        bar.local_entities(1), torch.tensor([[0, 1]])
    )  # the cell is the edge
    with pytest.raises(ValueError):
        bar.local_entities(2)  # a bar has no 2-D entity
