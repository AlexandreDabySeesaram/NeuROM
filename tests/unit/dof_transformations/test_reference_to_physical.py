"""WP4 push-forward (map-type) strategies: covariant / contravariant Piola + wiring.

The strategy ``reference_to_physical`` is tested in isolation against the defining invariants
(T-WP4.2 covariant preserves tangential inner products / contravariant preserves normal fluxes
w.r.t. ``R t``, on CCW *and* CW cells, with the signed ``det J``; T-WP4.3 the contravariant
autograd divergence equals ``(1/det J) div_hat``), then the wiring into point evaluation is checked
end-to-end with a synthetic contravariant element.
"""

import pytest
import torch

from neurom.elements import MapType, FiniteElement, P1_TRIANGLE
from neurom.dof_transformations import reference_to_physical

F64 = torch.float64
REL = 1e-12  # relative tolerance for the (float64, exact-algebra) comparisons; all expected != 0
# R: rotation by +90 deg (the symplectic form); satisfies J^T R J = det(J) R for any 2x2 J, which
# is exactly why the contravariant map preserves the flux through ``n = R t``.
R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=F64)

# A CCW cell (det J > 0) and a CW cell (det J < 0), both non-axis-aligned.
J_CCW = torch.tensor([[2.0, 0.3], [0.1, 1.5]], dtype=F64)  # det = 2.97
J_CW = torch.tensor([[2.0, 0.3], [1.5, 0.1]], dtype=F64)  # det = -0.25


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _detJ(J):
    return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]


@pytest.mark.parametrize("J", [J_CCW, J_CW], ids=["CCW", "CW"])
def test_identity_map_is_a_noop(J):
    v_ref = torch.tensor([0.7, -0.3])
    v_phys = reference_to_physical(v_ref, J, _detJ(J).reshape(1), MapType.IDENTITY)
    assert v_phys.numpy() == pytest.approx(v_ref.numpy(), rel=REL)


@pytest.mark.parametrize("J", [J_CCW, J_CW], ids=["CCW", "CW"])
def test_covariant_preserves_tangential_inner_product(J):
    """(T-WP4.2) J^{-T} maps covectors so that (J^{-T} v) . (J t) = v . t -- tangential/circulation."""
    v_ref = torch.tensor([0.7, -0.3])
    t_ref = torch.tensor([0.4, 1.1])  # a reference tangent; it pushes forward as J t
    v_phys = reference_to_physical(
        v_ref, J, _detJ(J).reshape(1), MapType.COVARIANT_PIOLA
    )
    t_phys = J @ t_ref
    assert float(v_phys @ t_phys) == pytest.approx(float(v_ref @ t_ref), rel=REL)


@pytest.mark.parametrize("J", [J_CCW, J_CW], ids=["CCW", "CW"])
def test_contravariant_preserves_normal_flux_signed(J):
    """(T-WP4.2) (J/det J) v . (R J t) = v . (R t) for the SIGNED det J, on CCW and CW cells."""
    v_ref = torch.tensor([0.7, -0.3])
    t_ref = torch.tensor([0.4, 1.1])
    detJ = _detJ(J)
    v_phys = reference_to_physical(v_ref, J, detJ.reshape(1), MapType.CONTRAVARIANT_PIOLA)
    t_phys = J @ t_ref  # the reference tangent pushes forward as J t
    n_phys = R @ t_phys  # physical normal = R * (pushed-forward tangent)
    n_ref = R @ t_ref  # reference normal = R * reference tangent
    assert float(v_phys @ n_phys) == pytest.approx(float(v_ref @ n_ref), rel=REL)


@pytest.mark.parametrize("J", [J_CCW, J_CW], ids=["CCW", "CW"])
def test_contravariant_ablation_absdet_flips_sign_on_cw(J):
    """(T-WP4.2 ablation) using |det J| gives sign(det J) * flux -> a -1 flux factor on CW cells."""
    v_ref = torch.tensor([0.7, -0.3])
    t_ref = torch.tensor([0.4, 1.1])
    detJ = _detJ(J)
    n_phys = R @ (J @ t_ref)  # physical normal = R * (pushed-forward tangent)
    n_ref = R @ t_ref  # reference normal = R * reference tangent
    v_phys_abs = torch.einsum("ij,j->i", J, v_ref) / detJ.abs()  # the |det J| variant (ablation)
    flux = v_phys_abs @ n_phys
    expected = torch.sign(detJ) * (v_ref @ n_ref)
    assert float(flux) == pytest.approx(float(expected), rel=REL)
    # Concretely: correct (+I) on the CCW cell, wrong (-I) on the CW cell.
    matches = float(flux) == pytest.approx(float(v_ref @ n_ref), rel=REL)
    assert matches == (detJ > 0).item()


def test_contravariant_divergence_equals_reference_over_detJ():
    """(T-WP4.3) autograd div of a contravariant-mapped field equals (1/det J) div_hat."""
    J, x0 = J_CCW, torch.tensor([0.5, -0.2])
    detJ = _detJ(J)
    J_inv = torch.linalg.inv(J)

    x = torch.tensor([[0.7, 0.4], [0.1, 0.9]], requires_grad=True)  # physical points
    xi = (x - x0) @ J_inv.T  # reference coords, xi = J^{-1} (x - x0)
    v_ref = torch.stack([xi[:, 0] ** 2, xi[:, 0] * xi[:, 1]], dim=-1)  # reference divergence = 3 xi0
    Jb = J.expand(x.shape[0], 2, 2)
    detb = detJ.reshape(1, 1).expand(x.shape[0], 1)
    v_phys = reference_to_physical(v_ref, Jb, detb, MapType.CONTRAVARIANT_PIOLA)

    div = sum(
        torch.autograd.grad(v_phys[:, i].sum(), x, create_graph=True)[0][:, i]
        for i in range(2)
    )
    expected = 3.0 * xi[:, 0] / detJ
    assert div.detach().numpy() == pytest.approx(expected.detach().numpy(), rel=REL)


def _single_triangle_setup():
    """A one-scalene-triangle mesh + an identity and a contravariant P1-vector element on it."""
    from neurom.meshes import Mesh, Topology
    from neurom.fields import Field
    from neurom.function_space import FunctionSpace
    from neurom.elements import VectorElement

    topo = Topology(torch.tensor([[0, 1, 2]]))
    pts = torch.tensor([[0.0, 0.0], [1.3, 0.2], [0.4, 1.1]])  # non-trivial J
    mesh = Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))
    basis, layout = P1_TRIANGLE.reference_basis, P1_TRIANGLE.dof_layout
    elem_id = FiniteElement(basis, layout, value_shape=(2,), map_type=MapType.IDENTITY)
    elem_pf = FiniteElement(
        basis, layout, value_shape=(2,), map_type=MapType.CONTRAVARIANT_PIOLA
    )
    dofs = torch.tensor([[1.0, 2.0], [-0.5, 0.3], [0.8, -1.2]])  # (n_scalar_dofs, 2)
    u_id = Field(FunctionSpace(topo, elem_id), dofs)
    u_pf = Field(FunctionSpace(topo, elem_pf), dofs)
    return mesh, topo, u_id, u_pf


def test_point_evaluation_applies_the_map():
    """The map-type is wired into point evaluation: a contravariant element's value is A_e-mapped."""
    from neurom.interpolation import PointWiseInterpolator
    from neurom.geometry import AffineSimplexMapping
    from neurom.samplings import ElementSampling

    mesh, _, u_id, u_pf = _single_triangle_setup()
    x = torch.tensor([[0.5, 0.4]])  # a point inside the triangle
    out_id = PointWiseInterpolator(mesh, u_id).at_position(
        x
    )  # the reference value (identity)
    out_pf = PointWiseInterpolator(mesh, u_pf).at_position(x)

    mapping = AffineSimplexMapping(mesh)
    eid = mesh.elements_at(x)
    J = mapping.jacobian_at(ElementSampling(x), eid).values
    det = mapping.det_jacobian.values[eid]
    expected = reference_to_physical(out_id, J, det, MapType.CONTRAVARIANT_PIOLA)
    assert out_pf.detach().numpy() == pytest.approx(expected.detach().numpy(), rel=REL)
    # the push-forward actually changed the value
    assert out_pf.detach().numpy() != pytest.approx(out_id.detach().numpy(), rel=REL)


def test_assembly_applies_the_map():
    """The map-type is wired into quadrature assembly too: assembled values are A_e-mapped."""
    from neurom.interpolation import QuadratureContext, QuadratureAssembly
    from neurom.dof_transformations.reference_to_physical import (
        reference_to_physical_at,
    )
    from neurom.quadratures import ThreePoints2D

    mesh, _, u_id, u_pf = _single_triangle_setup()
    ctx = QuadratureContext(mesh, ThreePoints2D())
    res_id = QuadratureAssembly(ctx, u_id).interpolate()
    res_pf = QuadratureAssembly(ctx, u_pf).interpolate()

    expected = reference_to_physical_at(
        res_id.u, ctx.mapping, ctx.interpolate.xi_back, MapType.CONTRAVARIANT_PIOLA
    )
    assert res_pf.u.values.detach().numpy() == pytest.approx(
        expected.values.detach().numpy(), rel=REL
    )
    assert res_pf.u.values.detach().numpy() != pytest.approx(
        res_id.u.values.detach().numpy(), rel=REL
    )
