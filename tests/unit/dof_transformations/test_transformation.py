import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.elements import P1_BAR, HERMITE, VectorElement
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
    return IsoparametricMapping1D(mesh)


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


def test_value_functional_on_function():
    f = lambda x: (x**2).squeeze(-1)
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    assert torch.allclose(
        ValueFunctional().on_function(f, x), torch.tensor([1.0, 4.0, 9.0], dtype=F64)
    )


def test_gradient_functional_on_function_is_derivative():
    f = lambda x: (x**2).squeeze(-1)  # f' = 2x
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
    assert torch.allclose(
        GradientFunctional().on_function(f, x), torch.tensor([2.0, 4.0, 6.0], dtype=F64)
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
