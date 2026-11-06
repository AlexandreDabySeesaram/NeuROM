import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import AffineSimplexMapping
from neurom.elements import P1_BAR, HERMITE, VectorElement, FiniteElement
from neurom.shape_functions import HermiteBeam
from neurom.dof_transformations import (
    Transformation,
    ValueFunctional,
    GradientFunctional,
)
from neurom.dof_transformations.transformation import _invert

F64 = torch.float64


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
        FiniteElement(HermiteBeam(), HERMITE.dof_layout), mapping
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
    J = torch.tensor([[[0.05]], [[0.15]], [[0.30]]], dtype=F64, requires_grad=True)  # (n_e,1,1)
    u = torch.ones(3, 4, 1, dtype=F64)

    def energy(g_e):
        q = torch.einsum("erp,ep...->er...", g_e(J), u)  # q = [1, J, 1, J] per element
        return (q**2).sum()  # S = sum_e (2 + 2 J_e^2)

    (g,) = torch.autograd.grad(energy(HERMITE.g_e_closed_form), J)
    assert g[..., 0, 0].numpy() == pytest.approx((4.0 * J[..., 0, 0]).detach().numpy(), rel=1e-12)

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


def test_invert_square_and_rectangular():
    # square -> inverse
    A = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=F64)
    assert torch.allclose(
        _invert(A), torch.tensor([[[0.5, 0.0], [0.0, 0.25]]], dtype=F64)
    )
    # rectangular (n_phys=3, n_ref=2) -> pinv of shape (1, 2, 3); pinv(L) @ L == I_2
    L = torch.randn(1, 3, 2, dtype=F64)
    M = _invert(L)
    assert M.shape == (1, 2, 3)
    assert torch.allclose(M @ L, torch.eye(2, dtype=F64).unsqueeze(0), atol=1e-10)
