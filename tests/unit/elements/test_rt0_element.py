"""WP9 stage C: the RT0_TRIANGLE element end-to-end (contravariant Piola + T_e facet signs).

The RT0 *basis* (flux duality V=I, divergence) is covered in
``tests/unit/shape_functions/test_raviart_thomas_0.py``; here we cover the assembled element: it is
a unisolvent contravariant vector element, and a field built from its edge-flux DOFs evaluates
correctly across cells -- which needs the contravariant Piola push-forward AND the per-cell facet
sign T_e = diag(sigma) that keeps the normal flux continuous over a shared edge. An ablation (drop
T_e) shows that continuity is exactly what the signs buy.
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import AffineSimplexMapping
from neurom.elements import P1_TRIANGLE, RT0_TRIANGLE, VectorElement, MapType
from neurom.dof_transformations import Transformation
from neurom.interpolation import PointWiseInterpolator

F64 = torch.float64


def _square():
    """Unit square as two triangles sharing the diagonal edge (0, 2)."""
    topo = Topology(torch.tensor([[0, 1, 2], [0, 2, 3]]))
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=F64)
    mesh = Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))
    return topo, mesh


def _constant_field_dofs(topo, pts, a, b):
    """Edge-flux DOFs of the constant field v = (a, b): u_i = v . (R t_i) on each global edge.

    The flux DOF uses the canonical (global) edge normal n = R t (unnormalised, t = x[hi] - x[lo]),
    so u_i = (a, b) . (t1, -t0). Constant fields lie in RT0, so these DOFs reproduce v exactly.
    """
    ev = topo.edge_vertices
    t = pts[ev[:, 1]] - pts[ev[:, 0]]
    n_canon = torch.stack([t[:, 1], -t[:, 0]], dim=-1)  # R t, R = -90 deg
    return (n_canon * torch.tensor([a, b], dtype=F64)).sum(dim=-1, keepdim=True)  # (n_edges, 1)


def test_rt0_element_is_a_unisolvent_contravariant_vector_element():
    """RT0_TRIANGLE: 3 flux DOFs dual to a 3-function vector basis (V=I), contravariant, scalar-coef."""
    e = RT0_TRIANGLE
    assert (e.n_dofs, e.n_ref) == (3, 3)  # unisolvent
    assert e.map_type == MapType.CONTRAVARIANT_PIOLA
    assert e.value_shape == () and e.n_components == 1  # scalar flux coeffs; vector value via basis
    assert e.V.numpy() == pytest.approx(torch.eye(3, dtype=e.V.dtype).numpy(), abs=1e-12)


def test_rt0_reproduces_a_constant_field_in_both_cells():
    """A constant field (in RT0) is recovered exactly in both triangles -- Piola + T_e + continuity.

    The shared diagonal has a single global flux DOF that the two cells traverse oppositely; only the
    T_e sign makes both read it consistently, so exact reproduction on *both* sides is the end-to-end
    check that the facet signs are right (see the ablation below).
    """
    topo, mesh = _square()
    pts = mesh.coordinates.full_values()
    a, b = 2.0, -3.0
    fld = Field(FunctionSpace(topo, RT0_TRIANGLE), _constant_field_dofs(topo, pts, a, b))

    x = torch.tensor([[0.6, 0.2], [0.2, 0.6]], dtype=F64)  # one point inside each triangle
    val = PointWiseInterpolator(mesh, fld).at_position(x)
    expected = torch.tensor([[a, b], [a, b]], dtype=F64)
    assert val.detach().numpy() == pytest.approx(expected.numpy(), rel=1e-12)


def test_dropping_T_e_breaks_the_reproduction():
    """(Ablation) without the facet signs T_e the same constant field is NOT reproduced.

    Confirms the previous test genuinely exercises T_e: a T_e-free transformation mis-signs the
    shared-edge DOF, so at least one cell returns the wrong constant vector.
    """
    topo, mesh = _square()
    pts = mesh.coordinates.full_values()
    a, b = 2.0, -3.0
    fld = Field(FunctionSpace(topo, RT0_TRIANGLE), _constant_field_dofs(topo, pts, a, b))

    pw = PointWiseInterpolator(mesh, fld)
    pw._dof_transformation = Transformation(RT0_TRIANGLE, AffineSimplexMapping(mesh))  # no T_e
    x = torch.tensor([[0.6, 0.2], [0.2, 0.6]], dtype=F64)
    val = pw.at_position(x).detach()
    expected = torch.tensor([[a, b], [a, b]], dtype=F64)
    # not both cells correct -> the whole array is not approx the constant field
    assert val.numpy() != pytest.approx(expected.numpy(), rel=1e-9)
