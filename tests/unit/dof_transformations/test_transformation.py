import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import AffineSimplexMapping, EntityGeometry
from neurom.elements import (
    P1_BAR,
    HERMITE,
    P1_TRIANGLE,
    P2_TRIANGLE,
    VectorElement,
    FiniteElement,
    MapType,
)
from neurom.shape_functions import CubicHermiteBar, QuadraticTriangle
from neurom.dof_transformations import (
    Transformation,
    ValueFunctional,
    GradientFunctional,
)
from neurom.dof_transformations.transformation import _invert

F64 = torch.float64


def _square_mesh():
    """Unit square as two triangles sharing the diagonal edge (0, 2) -- for the T_e facet-sign tests."""
    topo = Topology(torch.tensor([[0, 1, 2], [0, 2, 3]]))
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=F64)
    return Mesh(topo, Field(FunctionSpace(topo, VectorElement(P1_TRIANGLE, 2)), pts))


def _graded_mapping():
    # 3 graded cells on [0, 0.1, 0.4, 1.0] -> h = 0.1, 0.3, 0.6 -> J = h/2
    cell_vertices = torch.tensor([[0, 1], [1, 2], [2, 3]])
    points = torch.tensor([[0.0], [0.1], [0.4], [1.0]], dtype=F64)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, points))
    return AffineSimplexMapping(mesh)


def test_lagrange_is_identity_shortcircuit():
    T = Transformation(P1_BAR, _graded_mapping())
    assert T._identity is True
    u = torch.rand(3, 2, 1, dtype=F64)
    assert T.to_reference(u) is u  # untouched


def test_hermite_scales_slope_dofs_by_jacobian():
    T = Transformation(HERMITE, _graded_mapping())
    assert T._identity is False and T._geometry_dependent is True
    u = torch.ones(3, 4, 1, dtype=F64)
    out = T.to_reference(u)  # value DOFs unchanged, slopes scaled by J = h/2
    expected = torch.tensor(
        [[1.0, 0.05, 1.0, 0.05], [1.0, 0.15, 1.0, 0.15], [1.0, 0.30, 1.0, 0.30]],
        dtype=F64,
    ).unsqueeze(-1)
    assert torch.allclose(out, expected)


def test_hermite_closed_form_matches_generic_inversion():
    """(T-WP5.1) the closed-form Hermite M_e equals the generic assemble-L + invert, on a graded mesh."""
    mapping = _graded_mapping()
    T_closed = Transformation(HERMITE, mapping)  # uses g_e_closed_form
    T_generic = Transformation(  # same element, no closed form -> assemble L_e + invert
        FiniteElement(CubicHermiteBar(), HERMITE.dof_layout), mapping
    )
    assert T_closed._g_e_closed_form is not None and T_generic._g_e_closed_form is None
    torch.manual_seed(0)
    u = torch.randn(3, 4, 2, dtype=F64)  # (n_e, n_dofs, n_components)
    # (the generic path's autograd grad_N leaves a grad_fn -> detach before numpy)
    assert T_closed.to_reference(u).detach().numpy() == pytest.approx(
        T_generic.to_reference(u).detach().numpy(), rel=1e-12
    )


def test_hermite_closed_form_is_differentiable_in_the_jacobian():
    """(T-WP5.2) the closed-form G_e keeps M_e in the autograd graph of J_e, so the loss gradient
    reaches trainable node positions (the mapping already carries positions -> J_e in-graph, per the
    r-adaptivity examples). Here: sum of reference coeffs^2 with q = [1, J, 1, J] gives dS/dJ = 4 J.
    Detaching J inside G_e severs the dependence and the gradient vanishes (the ablation)."""
    J = torch.tensor(
        [[[0.05]], [[0.15]], [[0.30]]], dtype=F64, requires_grad=True
    )  # (n_e,1,1)
    u = torch.ones(3, 4, 1, dtype=F64)

    def energy(g_e):
        q = torch.einsum("erp,ep...->er...", g_e(J), u)  # q = [1, J, 1, J] per element
        return (q**2).sum()  # S = sum_e (2 + 2 J_e^2)

    (g,) = torch.autograd.grad(energy(HERMITE.g_e_closed_form), J)
    assert g[..., 0, 0].numpy() == pytest.approx(
        (4.0 * J[..., 0, 0]).detach().numpy(), rel=1e-12
    )

    # Ablation: detaching J inside G_e makes M_e constant -> the energy no longer depends on J at
    # all (no gradient path to node positions), which is exactly the bug the in-graph form avoids.
    assert not energy(lambda j: HERMITE.g_e_closed_form(j.detach())).requires_grad


def test_value_functional_on_function():
    # The component axis is kept explicit: a scalar f returning (n,) is normalised to (n, 1).
    f = lambda x: (x**2).squeeze(-1)
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    assert torch.allclose(
        ValueFunctional().on_function(f, x),
        torch.tensor([[1.0], [4.0], [9.0]], dtype=F64),
    )


def test_gradient_functional_on_function_is_derivative():
    # f' = 2x, returned with the spatial axis kept explicit: shape (n, 1), not (n,).
    f = lambda x: (x**2).squeeze(-1)
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    assert torch.allclose(
        GradientFunctional().on_function(f, x),
        torch.tensor([[2.0], [4.0], [6.0]], dtype=F64),
    )


def test_invert_square():
    A = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=F64)
    assert torch.allclose(
        _invert(A), torch.tensor([[[0.5, 0.0], [0.0, 0.25]]], dtype=F64)
    )


def test_invert_rejects_non_square_L():
    """(WP8 guard) a rectangular L_e raises rather than silently pseudo-inverting (a wrong M_e)."""
    L = torch.randn(1, 3, 2, dtype=F64)  # n_phys=3, n_ref=2
    with pytest.raises(NotImplementedError, match="non-square"):
        _invert(L)


def test_scalar_element_carries_no_T_e_even_with_entity_geometry():
    """A plain (IDENTITY) element gets no facet signs even if an EntityGeometry is supplied."""
    mesh = _square_mesh()
    T = Transformation(P2_TRIANGLE, AffineSimplexMapping(mesh), EntityGeometry(mesh))
    assert T._sign_facets is None  # map_type is IDENTITY -> T_e = I
    u = torch.ones(2, 6, 1, dtype=F64)
    assert torch.equal(T.to_reference(u), u)  # P2 is an identity transform, untouched


def test_contravariant_element_applies_facet_signs_as_T_e():
    """A direction-carrying (CONTRAVARIANT) element flips its facet DOFs by sigma; vertices unchanged.

    Uses a synthetic contravariant element (P2 basis/layout forced to CONTRAVARIANT) so G_e = I and
    to_reference reduces to exactly T_e = diag(sigma) -- isolating the sign factor (no flux
    functional needed). Expected signs are hand-derived, not read back from EntityGeometry.
    """
    mesh = _square_mesh()
    elem = FiniteElement(
        QuadraticTriangle(),
        P2_TRIANGLE.dof_layout,
        map_type=MapType.CONTRAVARIANT_PIOLA,
    )
    T = Transformation(elem, AffineSimplexMapping(mesh), EntityGeometry(mesh))
    assert T._sign_facets is not None

    out = T.to_reference(torch.ones(2, 6, 1, dtype=F64))  # G_e = I -> out = sigma (x) 1
    # 3 vertex DOFs unchanged (+1); then the 3 edge DOFs scaled by their orientation sign -- both
    # cells' local edge 2 (the (2,0)-reversed edge) is -1 (see the facet-signs ground truth).
    sign = torch.tensor([[1, 1, -1], [1, 1, -1]], dtype=F64)
    expected = torch.cat([torch.ones(2, 3, dtype=F64), sign], dim=1).unsqueeze(-1)
    assert out.numpy() == pytest.approx(expected.numpy(), rel=1e-12)
