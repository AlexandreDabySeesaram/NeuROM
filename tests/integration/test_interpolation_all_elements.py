"""§6.0 cross-element interpolation guard: every library element, through the full path.

For each element (registries ``EXACT_CASES`` for type 1, ``CONV_CASES`` for type 2):

* **type 1** -- interpolating a function *in* the element's space reproduces it exactly
  (to round-off, float64) at quadrature points (``QuadratureAssembly``) and at arbitrary
  points (``PointWiseInterpolator``), plus the derivatives the element is meant to reproduce;
  on the hard meshes (1-D graded, 2-D jittered scalene cells).
* **type 2** -- for an arbitrary field, the relative L2 error both converges at the element's
  rate *and* reaches a small absolute value on a modest mesh (the rate alone does not bound the
  error); both measured with a high-order, element-independent quadrature.
* **mutation self-checks** -- a deliberately broken interpolant must fail the type-1 check,
  proving the guard has teeth.

Adding an element = adding an ``ExactCase`` to ``EXACT_CASES`` (type 1) and a ``ConvCase`` to
``CONV_CASES`` (type 2); it is then tested automatically.
"""

import math

import pytest
import torch

from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.elements import (
    P1_BAR,
    P1_TRIANGLE,
    P2_TRIANGLE,
    HERMITE,
    DG0_BAR,
    DG0_TRIANGLE,
    VectorElement,
)
from neurom.quadratures import (
    TwoPoints1D,
    ThreePoints2D,
    GaussLegendre1D,
    GaussTriangle,
)
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    PointWiseInterpolator,
    interpolate,
)
from neurom.math import integrate, jacobian, second_derivative


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
# --- meshes
# ---------------------------------------------------------------------------


def _geometry_mesh(cell_vertices, points):
    """A mesh with a P1 coordinate field over ``cell_vertices`` (gdim = points.shape[1])."""
    topo = Topology(cell_vertices)
    gdim = points.shape[1]
    base = P1_BAR if cell_vertices.shape[1] == 2 else P1_TRIANGLE
    coords = Field(FunctionSpace(topo, VectorElement(base, gdim)), points)
    return Mesh(topo, coords)


def _bar_mesh(n, graded=False):
    """A 1-D mesh of ``n`` cells on ``[0, 1]``; ``graded`` uses Chebyshev (cosine) spacing.

    Uniform spacing is the easy case; the graded mesh clusters nodes near the ends so the cell
    size ``h`` varies strongly from cell to cell -- that non-constant ``h`` is what makes the
    gradient/second-derivative checks bite on geometry-dependent transforms (Hermite).
    """
    i = torch.arange(n + 1, dtype=torch.float64)
    x = 0.5 * (1 - torch.cos(math.pi * i / n)) if graded else i / n
    cells = torch.vstack([torch.arange(n), torch.arange(1, n + 1)]).T
    return _geometry_mesh(cells, x.unsqueeze(-1))


def _triangle_mesh(n, jitter=0.0, seed=0):
    """Structured unit-square triangle mesh: alternating diagonals, all cells wound CCW.

    ``jitter`` perturbs interior vertices by a seeded fraction of the cell size ``h = 1/n``, so
    the cells become generic scalene triangles with a non-trivial, cell-varying Jacobian (the
    interesting case for the geometric push-forward) while a refinement sequence stays a smooth,
    shape-regular family.
    """
    # Mesh size
    h = 1.0 / n

    # Define points on 2D grid
    xs, ys = torch.meshgrid(
        torch.linspace(0, 1, n + 1), torch.linspace(0, 1, n + 1), indexing="ij"
    )
    pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (n+1)^2, 2

    # Disrupt a bit the vertices positions to make it non-axis aligned
    if jitter:
        g = torch.Generator().manual_seed(seed)
        d = (torch.rand(pts.shape, generator=g) - 0.5) * 2 * jitter * h
        interior = (pts[:, 0] > 1e-9) & (pts[:, 0] < 1 - 1e-9)
        interior &= (pts[:, 1] > 1e-9) & (pts[:, 1] < 1 - 1e-9)
        pts = pts + d * interior.unsqueeze(-1)

    # Flat index
    def vid(i, j):
        return i * (n + 1) + j

    # Create triangles' connectivity
    cells = []
    for i in range(n):
        for j in range(n):
            a, b, c, d2 = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:  # alternate the diagonal
                tris = [[a, b, c], [a, c, d2]]
            else:
                tris = [[a, b, d2], [b, c, d2]]
            cells.extend(tris)
    cells = torch.tensor(cells)
    return _geometry_mesh(cells, pts)


# ---------------------------------------------------------------------------
# --- cases
# ---------------------------------------------------------------------------


class ExactCase:
    """A type-1 case: ``func`` lives *in* the element's space and must be reproduced exactly.

    Attributes:
        name: Human-readable id (used as the pytest parametrize id).
        element: The ``FiniteElement`` under test.
        dim: Topological dimension (1 bar, 2 triangle) -- selects the mesh/quadrature family.
        func: A function in the element's polynomial space (degree matched to the element, e.g. a
            cubic for Hermite), so interpolation is the identity.
        check_gradient: Also run the H1 gradient-exactness check (only for scalar elements meant
            to reproduce derivatives -- not DG0, not the vector element).
    """

    def __init__(self, name, element, dim, func, check_gradient=False):
        self.name = name
        self.element = element
        self.dim = dim
        self.func = func  # lives in the element's space (reproduced exactly)
        self.check_gradient = check_gradient  # also run the H1 gradient check


class ConvCase:
    """A type-2 case: ``func`` lies *outside* the element's space, so we measure how well it fits.

    Attributes:
        name: Human-readable id (used as the pytest parametrize id).
        element: The ``FiniteElement`` under test.
        dim: Topological dimension (1 bar, 2 triangle) -- selects the mesh/quadrature family.
        func: A function outside the space (a transcendental + high-degree mix).
        rate: Expected L2 convergence rate ``p+1`` for a degree-``p`` element.
        acc_n: Mesh resolution for the absolute-accuracy check.
        max_err: Relative L2 error the interpolant must reach at ``acc_n`` -- an *absolute* floor,
            since a correct convergence rate alone does not bound the error (a right slope can sit
            at a large constant). 1% for the approximating elements; looser for DG0, which is
            piecewise-constant (rate 1) and cannot reach 1% on an affordable mesh.
    """

    def __init__(self, name, element, dim, func, rate, acc_n, max_err):
        self.name = name
        self.element = element
        self.dim = dim
        self.func = func  # outside the element's space
        self.rate = rate  # expected L2 convergence rate p+1
        self.acc_n = acc_n  # mesh size for the absolute-accuracy check
        self.max_err = max_err  # max relative L2 error allowed at acc_n


def _poly1d(seed, deg):
    """A random 1-D polynomial of degree ``deg`` (seeded), as a callable ``X -> values``.

    Used to build functions that live exactly in a degree-``deg`` element's space: the seed makes
    the coefficients reproducible so a test failure is deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    c = torch.rand(deg + 1, generator=g)
    return lambda X: sum(c[k] * X[:, 0] ** k for k in range(deg + 1))


def _poly2d(seed, deg):
    """A random 2-D polynomial of *total* degree ``deg`` (seeded), as a callable ``X -> values``.

    The monomials ``x^i y^j`` with ``i + j <= deg`` span exactly the polynomial space of a
    degree-``deg`` triangle element (P1 -> deg 1, P2 -> deg 2), so its interpolant is exact.
    """
    g = torch.Generator().manual_seed(seed)
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    c = torch.rand(len(terms), generator=g)
    return lambda X: sum(
        c[k] * X[:, 0] ** i * X[:, 1] ** j for k, (i, j) in enumerate(terms)
    )


def _vec2d(seed, deg):
    """A 2-component vector field whose components are independent degree-``deg`` 2-D polynomials.

    The in-space exact function for ``VectorElement`` cases -- the two components use consecutive
    seeds so they differ, catching a component mix-up in the vectorised DOF layout.
    """
    fx, fy = _poly2d(seed, deg), _poly2d(seed + 1, deg)
    return lambda X: torch.stack([fx(X), fy(X)], dim=1)


# Fixed "arbitrary" functions for the type-2 convergence tests: deliberately *outside* every
# element's polynomial space (a transcendental part + a high-degree monomial) so no element
# reproduces them exactly and the L2 error genuinely decays at the element's rate.
_ARB_1D = lambda X: torch.cos(math.pi * X[:, 0]) + 3 * X[:, 0] ** 7
_ARB_2D = lambda X: (
    torch.cos(math.pi * X[:, 0]) * torch.sin(math.pi * X[:, 1])
    + 3 * X[:, 0] ** 7
    + X[:, 0] ** 2 * X[:, 1] ** 5
)
_ARB_VEC = lambda X: torch.stack(
    [_ARB_2D(X), _ARB_2D(torch.stack([X[:, 1], X[:, 0]], dim=1))], dim=1
)

# Type 1 -- a function IN each element's space, reproduced exactly. ``check_gradient`` is on only
# for scalar elements meant to reproduce derivatives (not DG0, not the vector element).
EXACT_CASES = [
    ExactCase("DG0_BAR", DG0_BAR, 1, _poly1d(0, 0)),
    ExactCase("P1_BAR", P1_BAR, 1, _poly1d(0, 1), check_gradient=True),
    ExactCase("HERMITE", HERMITE, 1, _poly1d(0, 3), check_gradient=True),
    ExactCase("DG0_TRIANGLE", DG0_TRIANGLE, 2, _poly2d(0, 0)),
    ExactCase("P1_TRIANGLE", P1_TRIANGLE, 2, _poly2d(0, 1), check_gradient=True),
    ExactCase("P2_TRIANGLE", P2_TRIANGLE, 2, _poly2d(0, 2), check_gradient=True),
    ExactCase("P1_TRIANGLE_vector", VectorElement(P1_TRIANGLE, 2), 2, _vec2d(0, 1)),
]

# Type 2 -- a function OUTSIDE each element's space, measured for rate + absolute accuracy. DG0 is
# piecewise-constant (rate 1): 1% needs an impractically fine mesh, so its max_err reflects its
# order while still catching a regression.
CONV_CASES = [
    ConvCase("DG0_BAR", DG0_BAR, 1, _ARB_1D, rate=1, acc_n=128, max_err=0.03),
    ConvCase("P1_BAR", P1_BAR, 1, _ARB_1D, rate=2, acc_n=32, max_err=0.01),
    ConvCase("HERMITE", HERMITE, 1, _ARB_1D, rate=4, acc_n=8, max_err=0.01),
    ConvCase("DG0_TRIANGLE", DG0_TRIANGLE, 2, _ARB_2D, rate=1, acc_n=32, max_err=0.10),
    ConvCase("P1_TRIANGLE", P1_TRIANGLE, 2, _ARB_2D, rate=2, acc_n=32, max_err=0.01),
    ConvCase("P2_TRIANGLE", P2_TRIANGLE, 2, _ARB_2D, rate=3, acc_n=8, max_err=0.01),
    ConvCase(
        "P1_TRIANGLE_vector",
        VectorElement(P1_TRIANGLE, 2),
        2,
        _ARB_VEC,
        rate=2,
        acc_n=32,
        max_err=0.01,
    ),
]


# ---------------------------------------------------------------------------
# --- helpers
# ---------------------------------------------------------------------------


def _type1_meshes(dim):
    """The hard meshes the type-1 exactness checks run on (the last one is the hardest).

    1-D: a uniform mesh plus a graded one (varying ``h``). 2-D: a single jittered mesh --
    distorted scalene cells with a non-trivial, cell-varying Jacobian, which is what stresses the
    geometric push-forward. The gradient checks deliberately pick ``[-1]`` (the hardest mesh).
    """
    if dim == 1:
        return [_bar_mesh(5), _bar_mesh(6, graded=True)]
    return [_triangle_mesh(3, jitter=0.25, seed=1)]


def _quad(dim):
    """The element-matched quadrature for the type-1 checks (exact-integration path).

    Low order on purpose: type 1 asks for exactness of functions in the space, which these rules
    integrate exactly -- unlike the high-order independent rule used for the type-2 error norms.
    """
    return TwoPoints1D() if dim == 1 else ThreePoints2D()


def _max_rel_error(actual, expected):
    """Max-norm error relative to the field's own scale: ``max|actual-expected| / max|expected|``.

    A relative measure is the right notion whenever the target is not identically zero (an
    absolute tolerance is meaningless without knowing the magnitude). One scalar serves both
    directions: the exact-reproduction checks require it near round-off, the mutation self-checks
    require it large. The ``clamp`` only guards the degenerate all-zero target.
    """
    denom = expected.abs().max().clamp_min(1e-300)
    return (actual - expected).abs().max().item() / denom.item()


# ---------------------------------------------------------------------------
# --- type 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EXACT_CASES, ids=lambda c: c.name)
def test_type1_exact_at_quadrature_points(case):
    """Interpolating a function already *in* the space reproduces it exactly at quad points.

    ``case.func`` lives in the element's polynomial space (e.g. a degree-1 polynomial for ``P1``,
    a cubic for ``HERMITE``), so interpolation is the identity: the nodal functionals recover the
    exact DOFs and ``QuadratureAssembly`` must evaluate back to ``f`` at every quadrature point,
    to round-off (relative error below ``1e-12`` under float64). This is the *representation* half
    of the guard -- it checks the whole assembly path (DOF map, basis tabulation, DOF
    transformation ``M_e``), not just the element in isolation, and on the hard meshes (graded
    1-D, jittered scalene cells in 2-D).
    """
    for mesh in _type1_meshes(case.dim):
        space = FunctionSpace(mesh.topology, case.element)
        u = Field(space, interpolate(space, mesh, case.func))
        res = QuadratureAssembly(
            QuadratureContext(mesh, _quad(case.dim)), u
        ).interpolate()
        x_q = res.x.values.reshape(-1, mesh.dim)
        f_q = case.func(x_q).reshape(res.u.values.shape)
        assert _max_rel_error(res.u.values, f_q) < 1e-12, case.name


@pytest.mark.parametrize("case", EXACT_CASES, ids=lambda c: c.name)
def test_type1_exact_pointwise(case):
    """Same exact reproduction, but at *arbitrary* interior points via ``PointWiseInterpolator``.

    Where the previous test samples the quadrature points baked into assembly, this one queries
    12 random points (kept away from the boundary, ``0.1 + 0.8*rand`` in 2-D, so every query
    lands unambiguously inside a cell). It exercises the *other* evaluation path -- point
    location + per-cell basis evaluation -- and must still return ``f`` for a function in the
    space (relative error below ``1e-11``; looser than the quad-point check because point location
    adds a pull-back solve). Together the two tests cover both ways the library evaluates a field.
    """
    for mesh in _type1_meshes(case.dim):
        space = FunctionSpace(mesh.topology, case.element)
        u = Field(space, interpolate(space, mesh, case.func))
        g = torch.Generator().manual_seed(7)
        if case.dim == 1:
            pts = torch.rand(12, 1, generator=g)
        else:  # points inside the unit square, away from the boundary
            pts = 0.1 + 0.8 * torch.rand(12, 2, generator=g)
        out = PointWiseInterpolator(mesh, u).at_position(pts)
        expected = case.func(pts).reshape(out.shape)
        assert _max_rel_error(out, expected) < 1e-11, case.name


def _grad_of(f, x):
    """Exact gradient of a scalar function ``f`` at points ``x``, via the library ``jacobian``.

    The reference the interpolant's gradient is compared against. Built on ``jacobian`` -- the same
    autograd path the library uses to differentiate fields -- rather than a hand-rolled
    ``autograd.grad`` call, so test and library differentiate identically. Returns shape
    ``(*x.shape[:-1], 1, dim)`` to match ``jacobian(res.x, res.u)``.
    """
    flat = x.reshape(-1, x.shape[-1]).detach().requires_grad_(True)
    return jacobian(flat, f(flat)).reshape(*x.shape[:-1], 1, x.shape[-1])


_GRAD_CASES = [c for c in EXACT_CASES if c.check_gradient]


@pytest.mark.parametrize("case", _GRAD_CASES, ids=lambda c: c.name)
def test_type1_gradient_exact(case):
    """The interpolant's *gradient* matches the exact gradient (H1 reproduction).

    Reproducing a function pointwise does not by itself guarantee the derivatives are right on a
    non-uniform mesh: the physical gradient involves the inverse cell Jacobian, so a wrong
    push-forward would still pass the value check yet fail here. Restricted to the ``check_gradient``
    cases (the ones whose space is meant to reproduce gradients), this compares
    ``jacobian(res.x, res.u)`` against the autograd gradient of ``f`` on the hard mesh (graded in
    1-D, jittered scalene cells in 2-D), to a relative error below ``1e-11``.
    """
    mesh = _type1_meshes(case.dim)[-1]  # graded (1-D) / jittered (2-D)
    space = FunctionSpace(mesh.topology, case.element)
    f = case.func
    u = Field(space, interpolate(space, mesh, f))
    res = QuadratureAssembly(QuadratureContext(mesh, _quad(case.dim)), u).interpolate()
    grad_h = jacobian(res.x, res.u)  # (n_e, n_q, 1, dim)
    grad_f = _grad_of(f, res.x.values)
    assert _max_rel_error(grad_h.values, grad_f) < 1e-11, case.name


def test_type1_hermite_second_derivative_exact():
    """Hermite reproduces a cubic's *second* derivative exactly (H2), even on a graded mesh.

    Hermite is the one element here with second-derivative accuracy, and its DOF transformation
    ``M_e`` is genuinely geometry-dependent (the slope DOFs scale with the physical cell size),
    so a mistake in ``M_e`` surfaces first in the curvature. This interpolates a cubic, takes
    ``second_derivative(res.x, res.u)``, and checks it against the twice-applied autograd
    derivative of ``f`` on a graded mesh (non-constant ``h``), to ``atol=1e-9``.
    """
    mesh = _bar_mesh(6, graded=True)
    space = FunctionSpace(mesh.topology, HERMITE)
    f = _poly1d(0, 3)
    u = Field(space, interpolate(space, mesh, f))
    res = QuadratureAssembly(QuadratureContext(mesh, TwoPoints1D()), u).interpolate()
    d2_h = second_derivative(res.x, res.u)  # (n_e, n_q, 1)
    x = res.x.values.reshape(-1, 1).detach().requires_grad_(True)
    (g,) = torch.autograd.grad(f(x).sum(), x, create_graph=True)
    (g2,) = torch.autograd.grad(g.sum(), x, create_graph=False)
    assert _max_rel_error(d2_h.values, g2.reshape(d2_h.values.shape)) < 1e-10


# ---------------------------------------------------------------------------
# --- type 2
# ---------------------------------------------------------------------------


def _l2_error(mesh, element, f, dim):
    """Relative L2 error ``||I_h f - f|| / ||f||`` of the interpolant, via a high-order rule.

    Integrates with the library ``integrate`` (which contracts the element, quadrature *and*
    component axes against the measure already carried by ``res.measure``), using a high-order
    element-independent quadrature so the measuring stick is far more accurate than the element.
    """
    space = FunctionSpace(mesh.topology, element)
    u = Field(space, interpolate(space, mesh, f))
    hi = GaussLegendre1D(8) if dim == 1 else GaussTriangle(8)
    res = QuadratureAssembly(QuadratureContext(mesh, hi), u).interpolate()
    f_q = f(res.x.values.reshape(-1, dim)).reshape(res.u.values.shape)
    w = res.measure.values  # (n_e, n_q, 1)
    err2 = integrate((res.u.values - f_q) ** 2 * w)
    nrm2 = integrate(f_q**2 * w)
    return (torch.sqrt(err2) / torch.sqrt(nrm2)).item()


@pytest.mark.parametrize("case", CONV_CASES, ids=lambda c: c.name)
def test_type2_convergence_rate(case):
    """For a function *outside* the space, the L2 interpolation error converges at the right rate.

    Type 1 checks exactness for functions the element can represent; this checks the approximation
    quality for one it cannot (``case.func``, a transcendental + high-degree mix). Standard FE
    theory gives ``||f - I_h f||_{L2} = O(h^{p+1})`` for a degree-``p`` element, i.e. the error
    should drop by ``2^{case.rate}`` per halving of ``h``. The empirical rate is read off the last
    two refinements (``log2(err[-2]/err[-1])``) and must clear ``case.rate - 0.25`` (a margin for
    pre-asymptotic noise). The error itself is measured with a high-order, element-independent rule
    (``GaussLegendre1D(8)`` / ``GaussTriangle(8)``) so the measuring stick never flatters the rate.
    """
    ns = [8, 16, 32] if case.dim == 1 else [4, 8, 16]
    f = case.func
    errs = []
    for n in ns:
        mesh = _bar_mesh(n) if case.dim == 1 else _triangle_mesh(n, jitter=0.15, seed=3)
        errs.append(_l2_error(mesh, case.element, f, case.dim))
    rate = math.log2(errs[-2] / errs[-1])
    assert rate >= case.rate - 0.25, f"{case.name}: rate {rate:.2f} < {case.rate}"


@pytest.mark.parametrize("case", CONV_CASES, ids=lambda c: c.name)
def test_type2_absolute_accuracy(case):
    """The interpolant reaches a small *absolute* error, not merely the right rate.

    A correct convergence rate is consistent with a large constant: an element can show the
    textbook slope yet sit at, say, 10% error -- which would be wrong. This pins the magnitude.
    On a single mesh sized per case (``case.acc_n``, kept small so the check stays fast), the
    relative L2 error of the out-of-space field must be at most ``case.max_err`` -- 1% for the
    approximating elements, looser for the piecewise-constant DG0 (see ``ConvCase.max_err``).
    """
    mesh = (
        _bar_mesh(case.acc_n)
        if case.dim == 1
        else _triangle_mesh(case.acc_n, jitter=0.15, seed=3)
    )
    err = _l2_error(mesh, case.element, case.func, case.dim)
    assert err <= case.max_err, f"{case.name}: rel L2 {err:.2e} > {case.max_err}"


# ---------------------------------------------------------------------------
# --- mutation self-checks
# ---------------------------------------------------------------------------

_MUTATION_MIN_ERROR = (
    1e-2  # a broken interpolant must be wrong by far more than round-off
)


def test_mutation_hermite_zeroed_slope_fails_on_graded_mesh():
    """Self-check: a *broken* Hermite interpolant (slope DOFs zeroed) is grossly wrong.

    A passing exactness test only means something if it can fail -- and fail *loudly*, not by a
    hair above the tolerance. Here the correct cubic interpolant is built and then the ``d1``
    (slope) DOFs are zeroed. We assert the resulting field is off by more than ``1%`` relative
    (it is ~4%), so the gap is real breakage, nowhere near the ``1e-12`` round-off floor the
    exactness check uses -- proving that check has genuine teeth on the geometry-sensitive slope
    DOFs, on a graded mesh where those DOFs carry real information.
    """
    f = _poly1d(0, 3)
    mesh = _bar_mesh(6, graded=True)
    space = FunctionSpace(mesh.topology, HERMITE)
    dofs = interpolate(space, mesh, f)
    dofs[space.locate(0, range(7), ["d1"])] = 0.0  # mutate: wrong slopes
    u = Field(space, dofs)
    res = QuadratureAssembly(QuadratureContext(mesh, TwoPoints1D()), u).interpolate()
    f_q = f(res.x.values.reshape(-1, 1)).reshape(res.u.values.shape)
    assert _max_rel_error(res.u.values, f_q) > _MUTATION_MIN_ERROR


def test_mutation_perturbed_dofs_fail_the_checker():
    """Self-check: perturbing *every* DOF of a correct P2 interpolant is grossly wrong.

    The 2-D counterpart of the mutation above, on the element with the richest DOF set in scope
    (P2: vertex + edge values). The exact quadratic interpolant is built, then every DOF is
    shifted by ``0.1``; the perturbed field must be off by more than ``1%`` relative (it is ~9%).
    This guards against a checker accidentally satisfied by any field (e.g. a tolerance that is
    too loose or a shape collapse), on a jittered mesh.
    """
    f = _poly2d(0, 2)
    mesh = _triangle_mesh(3, jitter=0.25, seed=1)
    space = FunctionSpace(mesh.topology, P2_TRIANGLE)
    dofs = interpolate(space, mesh, f)
    dofs = dofs + 0.1  # mutate every DOF
    u = Field(space, dofs)
    res = QuadratureAssembly(QuadratureContext(mesh, ThreePoints2D()), u).interpolate()
    f_q = f(res.x.values.reshape(-1, 2)).reshape(res.u.values.shape)
    assert _max_rel_error(res.u.values, f_q) > _MUTATION_MIN_ERROR
