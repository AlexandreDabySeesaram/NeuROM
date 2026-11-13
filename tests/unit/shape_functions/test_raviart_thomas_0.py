"""WP9 stage C: the RT0 vector reference basis (flux-nodal V = I, constant divergence -2)."""

import pytest
import torch

from neurom.shape_functions import RaviartThomas0Triangle
from neurom.reference_cell import TRIANGLE
from neurom.elements import EntityDim

F64 = torch.float64
R = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=F64)  # -90 deg rotation -> n = R t is OUTWARD (plan 3.1)


def test_n_ref_and_vector_shape():
    """RT0 reports n_ref = 3 (basis count, not gdim) and N tabulates as (*batch, n_ref, gdim)."""
    rt = RaviartThomas0Triangle()
    assert rt.n_ref == 3  # 3 basis functions, not gdim=2
    xi = torch.rand(4, 2, dtype=F64)
    assert rt.N(xi).shape == (4, 3, 2)  # (*batch, n_ref, gdim)


def test_flux_duality_is_identity():
    """l_i(H_j) = delta_ij for the cyclic-edge fluxes with n = R t, so the basis is flux-nodal (V=I)."""
    rt = RaviartThomas0Triangle()
    v = TRIANGLE.simplex.to(F64)  # (3, 2) reference vertices
    edges = TRIANGLE.local_entities(EntityDim.EDGE).tolist()  # cyclic (0,1),(1,2),(2,0)
    V = torch.zeros(3, 3, dtype=F64)
    for i, (a, b) in enumerate(edges):
        mid = 0.5 * (v[a] + v[b])
        n = R @ (v[b] - v[a])  # unnormalized: |n| = edge length folds in the measure
        V[i, :] = rt.N(mid) @ n  # flux of each basis function through edge i
    # identity target has zeros off-diagonal -> absolute tolerance
    assert V.numpy() == pytest.approx(torch.eye(3, dtype=F64).numpy(), abs=1e-12)


def test_divergence_is_constant_two():
    """Each RT0 basis function has constant reference divergence +2 (via autograd)."""
    rt = RaviartThomas0Triangle()
    xi = torch.tensor([[0.3, 0.4]], dtype=F64, requires_grad=True)
    H = rt.N(xi)  # (1, 3, 2)
    div = torch.stack(
        [
            sum(
                torch.autograd.grad(H[0, j, c], xi, retain_graph=True)[0][0, c]
                for c in range(2)
            )
            for j in range(3)
        ]
    )
    assert div.detach().numpy() == pytest.approx(
        (2.0 * torch.ones(3, dtype=F64)).numpy(), rel=1e-12
    )
