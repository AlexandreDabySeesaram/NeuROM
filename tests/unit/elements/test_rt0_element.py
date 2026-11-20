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
from neurom.geometry import AffineSimplexMapping, EntityGeometry
from neurom.elements import P1_TRIANGLE, RT0_TRIANGLE, VectorElement, MapType
from neurom.dof_transformations import Transformation
from neurom.dof_transformations.reference_to_physical import (
    reference_to_physical,
    reference_to_physical_at,
)
from neurom.samplings import ElementSampling
from neurom.interpolation import PointWiseInterpolator, interpolate

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


# ---------------------------------------------------------------------------
# --- WP9 acceptance tests (plan section 329): continuity, divergence, commuting
# ---------------------------------------------------------------------------


def _mixed_square():
    """Two triangles sharing the diagonal (0, 2), one wound CCW and one CW (det J < 0).

    cell 0 = [0,1,2] is CCW (det J = +1); cell 1 = [0,3,2] is CW (det J = -1) -- so the divergence /
    Piola checks exercise the *signed* det J on a reversed cell, as the plan requires.
    """
    topo = Topology(torch.tensor([[0, 1, 2], [0, 3, 2]]))
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=F64)
    mesh = Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))
    return topo, mesh


def _eval_in_cell(mesh, field, x, cell_id, with_signs=True):
    """Evaluate ``field`` at physical point ``x`` **as reconstructed by cell ``cell_id``**.

    Forces the element id (rather than ``elements_at``) so a point on a shared edge can be evaluated
    from either side; ``with_signs=False`` drops T_e (the facet-sign ablation).
    """
    elem = field.space.element
    mapping = AffineSimplexMapping(mesh)
    eg = EntityGeometry.for_element(elem, mesh) if with_signs else None
    T = Transformation(elem, mapping, eg)
    eid = torch.tensor([cell_id])
    u = T.to_reference(field.at_elements()[eid], eid)
    xi = mapping.inverse_map_at(ElementSampling(x), eid).values
    val = torch.einsum("en...,en...->e...", u, elem.reference_basis.N(xi))
    return reference_to_physical_at(
        ElementSampling(val), mapping, ElementSampling(x), elem.map_type, eid
    ).values


def _reference_divergences(element):
    """Constant reference divergence ``div_hat H_s`` of each basis function, via autograd (= 2 here).

    Read off the element's own basis (not hard-coded 2): another edge convention would change the
    signs, so the closed-form divergence below is built from these, per the plan.
    """
    xi = torch.tensor([[1.0 / 3, 1.0 / 3]], dtype=F64, requires_grad=True)
    H = element.reference_basis.N(xi)  # (1, n_ref, gdim)
    return torch.stack(
        [
            sum(torch.autograd.grad(H[0, s, c], xi, retain_graph=True)[0][0, c] for c in range(2))
            for s in range(element.n_ref)
        ]
    )


def _closed_form_div(mesh, field, d_ref):
    """Closed-form physical divergence per cell: ``(1/det J) sum_s sigma_{e,s} U_s div_hat H_s``."""
    detJ = AffineSimplexMapping(mesh).det_jacobian.values.squeeze(-1)  # (n_cells,) signed
    U = field.at_elements().squeeze(-1)  # (n_cells, n_dofs) flux DOFs
    sigma = EntityGeometry(mesh).facet_signs().to(U.dtype)  # (n_cells, n_dofs)
    return (sigma * U * d_ref).sum(dim=-1) / detJ  # (n_cells,)


def _autograd_div(mesh, field, cell_id, x):
    """Autograd divergence of the physical RT0 field inside cell ``cell_id`` at point ``x``."""
    elem = field.space.element
    mapping = AffineSimplexMapping(mesh)
    eid = torch.tensor([cell_id])
    q = Transformation(elem, mapping, EntityGeometry.for_element(elem, mesh)).to_reference(
        field.at_elements()[eid], eid
    ).squeeze(-1)  # (1, n_ref)
    nodes = mapping.x_nodes[cell_id]
    J = torch.stack([nodes[1] - nodes[0], nodes[2] - nodes[0]], dim=1)  # (2, 2) columns
    det = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]).reshape(1, 1)
    xg = x.clone().requires_grad_(True)
    xi = (xg - nodes[0]) @ torch.linalg.inv(J).T
    val = torch.einsum("en,eng->eg", q, elem.reference_basis.N(xi))
    v = reference_to_physical(val, J.unsqueeze(0), det, elem.map_type)  # (1, 2)
    return sum(torch.autograd.grad(v[0, c], xg, create_graph=True)[0][0, c] for c in range(2))


def test_rt0_normal_continuity_for_random_dofs_and_the_sign_ablation():
    """An arbitrary RT0 field is normal-continuous across a shared edge; without T_e the normal jumps.

    Continuity for *reproduced* fields is weaker than for arbitrary DOFs (plan section 267), so this
    uses random flux DOFs. The normal component agrees from both cells (H(div) conformity); the
    tangential component may jump. Dropping the facet signs breaks it with an order-1 jump.
    """
    topo, mesh = _square()  # both CCW -> the shared diagonal has OPPOSITE sigma in the two cells
    sp = FunctionSpace(topo, RT0_TRIANGLE)
    g = torch.Generator().manual_seed(1)
    fld = Field(sp, torch.randn(sp.n_scalar_dofs, 1, generator=g, dtype=F64))

    x = torch.tensor([[0.5, 0.5]], dtype=F64)  # midpoint of the shared diagonal (0, 2)
    n = torch.tensor([[1.0], [-1.0]], dtype=F64) / (2.0 ** 0.5)  # unit edge normal
    t = torch.tensor([[1.0], [1.0]], dtype=F64) / (2.0 ** 0.5)  # unit edge tangent
    v0, v1 = _eval_in_cell(mesh, fld, x, 0), _eval_in_cell(mesh, fld, x, 1)
    # normal component continuous across the edge ...
    assert float(v0 @ n) == pytest.approx(float(v1 @ n), rel=1e-10)
    # ... while the tangential component genuinely jumps (so the field is non-trivial, not constant)
    assert abs(float(v0 @ t) - float(v1 @ t)) > 0.1

    # ablation: without T_e the normal component flips in one cell -> an order-1 jump
    a0 = _eval_in_cell(mesh, fld, x, 0, with_signs=False)
    a1 = _eval_in_cell(mesh, fld, x, 1, with_signs=False)
    assert abs(float(a0 @ n) - float(a1 @ n)) > 0.1


def test_rt0_divergence_closed_form_matches_autograd_on_ccw_and_cw_cells():
    """The declared-divergence closed form equals the autograd divergence, on a CCW and a CW cell.

    ``div v|_e = (1/det J_e) sum_s sigma_{e,s} U_s div_hat H_s`` -- built from the element's own
    reference divergences (= 2) and the signed det J. Checked against autograd on both cells, so the
    sign of det J (negative on the CW cell) is exercised.
    """
    topo, mesh = _mixed_square()
    detJ = AffineSimplexMapping(mesh).det_jacobian.values.squeeze(-1)
    assert detJ[0] > 0 and detJ[1] < 0  # cell 0 CCW, cell 1 CW

    d_ref = _reference_divergences(RT0_TRIANGLE)
    assert d_ref.detach().numpy() == pytest.approx([2.0, 2.0, 2.0], rel=1e-12)  # branch convention

    sp = FunctionSpace(topo, RT0_TRIANGLE)
    g = torch.Generator().manual_seed(5)
    fld = Field(sp, torch.randn(sp.n_scalar_dofs, 1, generator=g, dtype=F64))
    closed = _closed_form_div(mesh, fld, d_ref)
    probes = [torch.tensor([[0.6, 0.5]], dtype=F64), torch.tensor([[0.3, 0.6]], dtype=F64)]
    for e in range(2):
        auto = float(_autograd_div(mesh, fld, e, probes[e]))
        assert closed[e].item() == pytest.approx(auto, rel=1e-10)


def test_rt0_commuting_property_on_ccw_and_cw_cells():
    """``div(I_h f) = `` cell-mean of ``div f`` -- the H(div) commuting diagram, on CCW and CW cells.

    For an affine ``f`` the one-point edge flux is exact, so the identity holds to round-off: an affine
    ``f`` has constant divergence, and ``div(I_h f)`` (constant per RT0 cell) must equal it. ``f`` is
    outside RT0 (independent component gradients), so this is not mere reproduction.
    """
    topo, mesh = _mixed_square()
    f = lambda X: torch.stack([2 * X[:, 0] + X[:, 1], X[:, 0] + 3 * X[:, 1]], dim=1)  # div f = 5
    div_f = 5.0

    sp = FunctionSpace(topo, RT0_TRIANGLE)
    I_h = Field(sp, interpolate(sp, mesh, f))
    div_Ih = _closed_form_div(mesh, I_h, _reference_divergences(RT0_TRIANGLE))
    assert div_Ih.detach().numpy() == pytest.approx([div_f, div_f], rel=1e-10)
