r"""Exactness and invariants of the high-order Gauss rules (GaussLegendre1D, GaussTriangle).

Every assertion here checks a *closed form*; those closed forms are derived in
``docs/quadrature_derivation.pdf`` -- §2 (Gauss-Legendre), §3 (Duffy triangle), and §5, which
collects the identities the tests compare against. Governing facts:

* **Gauss-Legendre, degree 2n-1.** An ``n``-point rule has ``2n`` free parameters
  :math:`(\xi_i, w_i)` matched to the ``2n`` monomials :math:`1,\dots,x^{2n-1}`, so it is exact
  for every polynomial of degree :math:`\le 2n-1`, and ``x^{2n}`` is the first degree it misses
  (PDF §2, §5.2).
* **Duffy triangle, total degree 2n-2.** With ``n`` points per collapsed direction, a monomial
  of total degree ``d`` becomes degree ``d+1`` in the collapsed coordinate (the Duffy Jacobian
  ``1-v``), so the per-direction budget ``2n-1`` must cover ``d+1``, giving ``d <= 2n-2``
  (PDF §3.3).

Reference moments compared against (PDF §5.1):

* bar :math:`[-1,1]`:  :math:`\int_{-1}^{1} x^k\,dx = 0` (``k`` odd), :math:`2/(k+1)` (``k`` even);
* reference triangle:  :math:`\int_{\hat K} x^i y^j\,dA = \dfrac{i!\,j!}{(i+j+2)!}` (a
  Dirichlet / Beta integral).
"""

import math

import pytest
import torch

from neurom.quadratures import GaussLegendre1D, GaussTriangle
from neurom.geometry.barycentric_to_reference import barycentric_to_reference


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _ref_points(quad):
    """Map a rule's barycentric points to reference coordinates (``sum_n lambda_n v_n``)."""
    return barycentric_to_reference(quad.points_barycentric, quad.reference_cell)


# ----------------------------------------------------------------- invariants


@pytest.mark.parametrize(
    "factory",
    [lambda: GaussLegendre1D(3), lambda: GaussLegendre1D(6), lambda: GaussTriangle(4)],
)
def test_weights_sum_to_reference_measure(factory):
    r"""The weights integrate the constant 1 exactly: :math:`\sum_q \omega_q = |\hat K|`.

    This is the quadrature's defining normalization -- it is what makes the physical integral
    :math:`\sum_q \omega_q |\det J_e|` reduce to :math:`|K_e|` on an affine cell (PDF §1, eq. 2).
    ``|K_hat|`` is 2 for the bar ``[-1,1]`` and 1/2 for the reference triangle.
    """
    quad = factory()  # built under the float64 fixture (not at collection time)
    assert quad.weights_ref.sum().item() == pytest.approx(
        quad.reference_cell.measure.item(), abs=1e-12
    )


@pytest.mark.parametrize(
    "factory", [lambda: GaussLegendre1D(5), lambda: GaussTriangle(5)]
)
def test_points_are_barycentric(factory):
    r"""Stored points are valid barycentric coordinates: components in ``[0,1]`` summing to 1.

    For the Duffy rule this holds by construction: with :math:`(u,v)\in[0,1]^2`,
    :math:`\xi=u(1-v)\ge0`, :math:`\eta=v\ge0`, and :math:`\xi+\eta=u+v(1-u)\le1`, so
    :math:`\lambda_0=1-\xi-\eta\ge0` (PDF §3.1). Validity here guarantees every quadrature point
    lies inside the reference cell.
    """
    p = factory().points_barycentric
    assert (p >= -1e-12).all() and (p <= 1 + 1e-12).all()
    assert torch.allclose(p.sum(dim=1), torch.ones(p.shape[0]), atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_agnostic(dtype):
    """Rules build their tensors in the *ambient* torch dtype (no float32 leak from import time).

    The NumPy Gauss nodes are cast to ``torch.get_default_dtype()`` at construction, so a rule
    built under float64 is float64 end to end -- required for the 1e-12 exactness checks below.
    """
    torch.set_default_dtype(dtype)
    assert GaussLegendre1D(4).points_barycentric.dtype == dtype
    assert GaussTriangle(4).weights_ref.dtype == dtype


# ----------------------------------------------------------------- 1-D exactness


@pytest.mark.parametrize("n", [2, 3, 5])
def test_gauss_legendre_exact_to_degree_2n_minus_1(n):
    r"""Gauss(n) integrates every monomial up to degree ``2n-1`` exactly.

    Checked against the elementary moment
    :math:`\int_{-1}^{1} x^k\,dx = \big[x^{k+1}/(k+1)\big]_{-1}^{1}
    = \big(1-(-1)^{k+1}\big)/(k+1)`, i.e. ``0`` for odd ``k`` (antisymmetry) and ``2/(k+1)`` for
    even ``k`` (PDF §5.1). Degree ``2n-1`` is the maximum a rule with ``2n`` free parameters can
    reach (PDF §2).
    """
    q = GaussLegendre1D(n)
    xi = _ref_points(q).squeeze(-1)  # on [-1, 1]
    for k in range(2 * n):  # degrees 0 .. 2n-1
        got = (xi**k * q.weights_ref).sum().item()
        exact = 0.0 if k % 2 == 1 else 2.0 / (k + 1)  # ∫_{-1}^{1} x^k dx
        assert got == pytest.approx(exact, abs=1e-12), f"degree {k}"


def test_gauss_legendre_not_exact_beyond():
    r"""Degree ``2n`` is the *first* degree Gauss(n) misses -- the exactness is tight, not lucky.

    The rule's error on a degree-``2n`` polynomial is proportional to its ``2n``-th derivative,
    which is a nonzero constant and does not cancel (the error term is sign-definite; PDF §5.2).
    Concretely the 2-point rule (exact only to degree 3) integrates ``x^4`` to ``2/9`` rather
    than the true ``2/5``.
    """
    q = GaussLegendre1D(2)
    xi = _ref_points(q).squeeze(-1)
    got = (xi**4 * q.weights_ref).sum().item()
    assert abs(got - 2.0 / 5.0) > 1e-3


# ----------------------------------------------------------------- 2-D exactness


@pytest.mark.parametrize("n", [3, 5])
def test_gauss_triangle_exact_to_total_degree_2n_minus_2(n):
    r"""GaussTriangle(n) integrates every ``x^i y^j`` with ``i+j <= 2n-2`` exactly.

    Checked against the reference-triangle moment
    :math:`\int_{\hat K} x^i y^j\,dA = \dfrac{i!\,j!}{(i+j+2)!}`, which follows from the iterated
    integral :math:`\int_0^1 y^j\!\int_0^{1-y} x^i\,dx\,dy` and the Beta integral
    :math:`\int_0^1 t^{a-1}(1-t)^{b-1}\,dt = (a-1)!(b-1)!/(a+b-1)!` (PDF §5.1). The exactness
    bound ``2n-2`` is derived in PDF §3.3 (the Duffy Jacobian costs one degree).
    """
    q = GaussTriangle(n)
    xy = _ref_points(q)  # (Nq, 2) on the reference triangle
    x, y = xy[:, 0], xy[:, 1]
    for i in range(2 * n - 1):
        for j in range(2 * n - 1 - i):  # total degree i+j <= 2n-2
            got = (x**i * y**j * q.weights_ref).sum().item()
            exact = math.factorial(i) * math.factorial(j) / math.factorial(i + j + 2)
            assert got == pytest.approx(exact, abs=1e-12), f"x^{i} y^{j}"
