"""2-D directional-derivative functional via :math:`J^{-\\top}` (T-WP2.3).

``GradientFunctional(direction=d)`` computes :math:`\\mathbf d\\cdot\\nabla` in any dimension: on a
user function via autograd, and on the reference basis by pushing the reference gradient to physical
space with :math:`J_e^{-\\top}`. These tests check both against independent oracles on a
non-axis-aligned triangle (so :math:`J` is non-trivial and non-diagonal).
"""

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import AffineSimplexMapping
from neurom.elements import P1_BAR, P1_TRIANGLE, VectorElement
from neurom.dof_transformations import GradientFunctional
from neurom.shape_functions import LinearBar, LinearTriangle

F64 = torch.float64


def _one_bar_mapping():
    """A single bar [0, 2] (length 2, so J = h/2 = 1)."""
    cell_vertices = torch.tensor([[0, 1]])
    points = torch.tensor([[0.0], [2.0]], dtype=F64)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    mesh = Mesh(topology, Field(geometry, points))
    return AffineSimplexMapping(mesh)


def _one_triangle_mapping():
    """A single non-axis-aligned triangle a=(0,0), b=(2,0), c=(1,3) -> non-diagonal J."""
    cell_vertices = torch.tensor([[0, 1, 2]])
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]], dtype=F64)
    topology = Topology(cell_vertices)
    geometry = FunctionSpace(topology, VectorElement(P1_TRIANGLE, 2))
    mesh = Mesh(topology, Field(geometry, points))
    return AffineSimplexMapping(mesh), points


def test_on_function_matches_analytic_partials():
    """d.grad f with d = e_x / e_y recovers the analytic partials of a nonlinear f."""
    f = lambda x: (x[:, 0:1] ** 2) * x[:, 1:2]  # f = x0^2 x1 -> df/dx0 = 2 x0 x1, df/dx1 = x0^2
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=F64)

    dfx = GradientFunctional(direction=[1.0, 0.0]).on_function(f, x)
    dfy = GradientFunctional(direction=[0.0, 1.0]).on_function(f, x)

    assert torch.allclose(dfx, (2.0 * x[:, 0:1] * x[:, 1:2]))
    assert torch.allclose(dfy, (x[:, 0:1] ** 2))


def test_l_rows_pushes_reference_gradient_with_J_inv_transpose():
    """(L_e)_ia = d.grad N_a reproduces d.grad of any linear p via the FE-gradient identity.

    For p in the P1 space, sum_a p(x_a) grad N_a(xi_i) == grad p exactly, independent of the
    J^{-T} formula -- so sum_a p(x_a) (L_e)_ia must equal d.grad p. With p = x the e_x derivative
    is 1 and the e_y derivative 0; with p = y they swap.
    """
    mapping, points = _one_triangle_mapping()
    basis = LinearTriangle()
    xi = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=F64)  # reference vertices

    L_ex = GradientFunctional(direction=[1.0, 0.0]).l_rows(basis, mapping, xi, None, 1)
    L_ey = GradientFunctional(direction=[0.0, 1.0]).l_rows(basis, mapping, xi, None, 1)

    p_x = points[:, 0]  # nodal values of p(x, y) = x
    p_y = points[:, 1]  # nodal values of p(x, y) = y

    # sum_a p(x_a) (L_e)_{i a}  -> directional derivative of the interpolant at each xi_i
    assert torch.allclose(L_ex[0] @ p_x, torch.ones(3, dtype=F64))   # d/dx of x = 1
    assert torch.allclose(L_ex[0] @ p_y, torch.zeros(3, dtype=F64))  # d/dx of y = 0
    assert torch.allclose(L_ey[0] @ p_x, torch.zeros(3, dtype=F64))  # d/dy of x = 0
    assert torch.allclose(L_ey[0] @ p_y, torch.ones(3, dtype=F64))   # d/dy of y = 1


def test_1d_direction_reduces_to_plain_derivative():
    """In 1-D, ``direction=[1.0]`` (the general J^{-T} path) equals ``direction=None`` (d/dx).

    Both ``on_function`` (f' of a known f) and ``l_rows`` (the basis derivative over the bar)
    must agree, confirming the generalized path specializes correctly to the 1-D Hermite case.
    """
    mapping = _one_bar_mapping()
    basis = LinearBar()
    xi = torch.tensor([[-1.0], [1.0]], dtype=F64)  # reference bar endpoints

    plain = GradientFunctional(direction=None)
    directed = GradientFunctional(direction=[1.0])

    # l_rows agree
    assert torch.allclose(
        plain.l_rows(basis, mapping, xi, None, 1),
        directed.l_rows(basis, mapping, xi, None, 1),
    )

    # on_function agree, and equal the analytic derivative f'(x) = 2x
    f = lambda x: x**2
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    expected = 2.0 * x  # (n, 1)
    assert torch.allclose(plain.on_function(f, x), expected)
    assert torch.allclose(directed.on_function(f, x), expected)


def test_l_rows_raises_in_2d_without_a_direction():
    """``direction=None`` is the 1-D plain derivative; in 2-D it is ambiguous and must raise."""
    mapping, _ = _one_triangle_mapping()
    basis = LinearTriangle()
    xi = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=F64)
    with pytest.raises(NotImplementedError, match="1-D only"):
        GradientFunctional(direction=None).l_rows(basis, mapping, xi, None, 1)


def test_reference_row_raises_for_a_directional_functional():
    """The reference duality row for a directional derivative is deferred to WP9 -- it must raise."""
    basis = LinearTriangle()
    xi = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=F64)
    with pytest.raises(NotImplementedError, match="directional"):
        GradientFunctional(direction=[1.0, 0.0]).reference_row(basis, xi)


def test_direction_gradients_flow_to_node_positions():
    """l_rows is in-graph w.r.t. node positions (r-adaptivity): a grad path exists."""
    mapping, _ = _one_triangle_mapping()
    mapping._mesh.coordinates.values.requires_grad_(True)
    mapping.update()
    basis = LinearTriangle()
    xi = torch.tensor([[0.3, 0.3]], dtype=F64)
    L = GradientFunctional(direction=[1.0, 0.0]).l_rows(basis, mapping, xi, None, 1)
    L.sum().backward()
    assert mapping._mesh.coordinates.values.grad is not None
