import pytest
import torch

from neurom.decompositions import (
    Axis,
    CPPGD,
    PolynomialNLPGD,
    total_degree_exponents,
    uniform_exponents,
)
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import NoConstraint
from neurom.field_layout import FieldLayout
from neurom.integrate import integrate
from neurom.interpolation import IntegrationDomain
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator

torch.set_default_dtype(torch.float32)


def make_axis(name="space", n=5, lo=0.0, hi=10.0, dim=1):
    coords = torch.linspace(lo, hi, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(nodes, elements)
    positions = Field(name=f"{name}_positions", topology=topology, values=coords)
    sf = LinearSegment()
    return Axis(
        name=name,
        nodes_positions=positions,
        sf=sf,
        mapping=IsoparametricMapping1D(sf),
        quad=TwoPoints1D(),
        constraint=NoConstraint(),
        init_values=torch.zeros(n, dim),
    )


def make_two_axes():
    return [
        make_axis(name="space", n=5, lo=0.0, hi=10.0),
        make_axis(name="E", n=4, lo=100.0, hi=1000.0),
    ]


# Values small enough that squares and cubes stay well inside float32 range.
MONOM_VALUES = {
    (0, 0): torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6]),
    (0, 1): torch.tensor([0.2, 0.3, 0.4, 0.5]),
    (1, 0): torch.tensor([0.1, -0.1, 0.2, 0.05, -0.05]),
    (1, 1): torch.tensor([-0.2, 0.1, 0.0, 0.3]),
}

QUERY_X = torch.tensor([2.5, 5.0, 7.5])
QUERY_E = torch.tensor([400.0, 700.0])


def seed_monoms(model, n_modes):
    with torch.no_grad():
        for m in range(n_modes):
            for k in range(2):
                model.monoms[m][k].values_reduced.copy_(
                    MONOM_VALUES[(m, k)].unsqueeze(-1)
                )


def interpolate(model, axes, m, k, coords):
    pwi = PointWiseInterpolator(
        axes[k].mesh, axes[k].sf, model.monoms[m][k], axes[k].mapping
    )
    return pwi.at_position(coords).reshape(-1)


def separable_energy(layout, directory, terms):
    """A minimal separable energy, linear in the field: ``int u dOmega``.

    Not physical -- it exists so the tests exercise the real gradient path
    (through the ``FieldLayout`` the ``IntegrationDomain`` fills), which is the
    only differentiable one; ``evaluate``/``assemble`` are detached by
    ``PointWiseInterpolator``. It is also the smallest consumer of
    :meth:`PolynomialNLPGD.polynomial_directory`, so it doubles as a check that
    the seam is usable.

    Args:
        layout (FieldLayout): filled layout.
        directory (dict): axis name -> monom names per mode.
        terms (list): ``polynomial_directory()`` output, or ``None`` for the
            plain CP form (leading terms only).
    """
    axis_names = list(directory)
    if terms is None:
        n_modes = len(directory[axis_names[0]])
        terms = [(m, (1,) * len(axis_names), None) for m in range(n_modes)]

    total = 0.0
    for mode, exponents, coefficient in terms:
        contribution = 1.0 if coefficient is None else coefficient
        for k, axis_name in enumerate(axis_names):
            r = layout[directory[axis_name][mode]]
            contribution = contribution * integrate(r.u ** exponents[k] * r.measure)
        total = total + contribution
    return total


def fill_layout(deco):
    """Register + interpolate every active monom, returning the filled layout."""
    layout = FieldLayout()
    deco.register_into(layout)
    IntegrationDomain(deco.assemblies()).interpolate_all(layout)
    return layout


# --------------------------------------------------------------------------
# 1. Exponent-set builders
# --------------------------------------------------------------------------


def test_uniform_exponents_shape_and_content():
    exps = uniform_exponents(3, 4)
    assert exps.shape == (3, 3)  # p = 2, 3, 4
    assert exps.dtype == torch.long
    assert torch.equal(exps, torch.tensor([[2, 2, 2], [3, 3, 3], [4, 4, 4]]))
    # Independent of d in row count.
    assert uniform_exponents(7, 4).shape == (3, 7)


def test_uniform_exponents_excludes_leading_and_rejects_max_power_one():
    exps = uniform_exponents(4, 3)
    assert not bool((exps == 1).all(dim=1).any())
    with pytest.raises(ValueError, match="max_power >= 2"):
        uniform_exponents(4, 1)


def test_total_degree_exponents_content():
    exps = total_degree_exponents(2, 4)
    # lambda_j >= 1, 2 < sum <= 4 -> (1,2) (2,1) (1,3) (2,2) (3,1)
    got = {tuple(int(v) for v in row) for row in exps}
    assert got == {(1, 2), (2, 1), (1, 3), (2, 2), (3, 1)}
    assert exps.dtype == torch.long


def test_total_degree_exponents_invariants():
    exps = total_degree_exponents(3, 6)
    assert bool((exps >= 1).all())
    assert bool((exps.sum(dim=1) <= 6).all())
    assert not bool((exps == 1).all(dim=1).any())  # leading term absent
    assert torch.unique(exps, dim=0).shape[0] == exps.shape[0]  # no duplicates


def test_total_degree_exponents_empty_set_raises():
    with pytest.raises(ValueError, match="max_total > 5"):
        total_degree_exponents(5, 5)
    with pytest.raises(ValueError, match="is empty"):
        total_degree_exponents(5, 3)


# --------------------------------------------------------------------------
# 2. Reduction to CPPGD when C = 0
# --------------------------------------------------------------------------


def test_zero_coefficients_reduce_to_cppgd():
    axes = make_two_axes()
    poly = PolynomialNLPGD(
        axes=axes, n_modes_max=2, exponents=uniform_exponents(2, 3), n_modes_ini=2
    )
    cp = CPPGD(axes=make_two_axes(), n_modes_max=2, n_modes_ini=2)
    seed_monoms(poly, 2)
    seed_monoms(cp, 2)

    assert torch.allclose(
        poly.assemble([QUERY_X, QUERY_E]), cp.assemble([QUERY_X, QUERY_E])
    )

    pts = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 900.0])], dim=1)
    assert torch.allclose(poly.evaluate(pts), cp.evaluate(pts))


def test_zero_coefficients_do_not_perturb_monom_gradients():
    """A zero C must contribute nothing to the *monom* gradients either.

    This is the claim that makes `requires_grad`-only lifecycle control
    sufficient (no `active` flag): with C = 0 and unfrozen, the polynomial terms
    are numerically absent. Checked, not assumed -- and checked on the real
    (layout) gradient path, since evaluate/assemble are detached.
    """
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    cp = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    seed_monoms(poly, 1)
    seed_monoms(cp, 1)
    poly.unfreeze_mode_coefficients(0)

    separable_energy(
        fill_layout(poly), poly.directory(), poly.polynomial_directory()
    ).backward()
    separable_energy(fill_layout(cp), cp.directory(), None).backward()

    for k in range(2):
        g_poly = poly.monoms[0][k].values_reduced.grad
        g_cp = cp.monoms[0][k].values_reduced.grad
        assert g_poly is not None and g_cp is not None
        assert torch.allclose(g_poly, g_cp, atol=1e-6)
    # The coefficient row itself does see a gradient -- it is not degenerate.
    assert poly.coefficients[0].grad is not None
    assert float(poly.coefficients[0].grad.abs().max()) > 0.0


# --------------------------------------------------------------------------
# 3. Correctness with C != 0
# --------------------------------------------------------------------------


def brute_force_evaluate(model, axes, exps, coeffs, x, e):
    """Independent sum over (mode, lambda), built straight from the interpolator."""
    total = torch.zeros_like(x)
    for m in range(model.n_modes_truncated):
        s = interpolate(model, axes, m, 0, x)
        g = interpolate(model, axes, m, 1, e)
        total = total + s * g
        for t, lam in enumerate(exps):
            total = total + coeffs[m][t] * s ** int(lam[0]) * g ** int(lam[1])
    return total


def test_evaluate_matches_brute_force_with_nonzero_coefficients():
    axes = make_two_axes()
    exps = total_degree_exponents(2, 4)
    poly = PolynomialNLPGD(axes=axes, n_modes_max=2, exponents=exps, n_modes_ini=2)
    seed_monoms(poly, 2)
    coeffs = [
        torch.linspace(0.5, 1.5, poly.n_terms),
        torch.linspace(-1.0, 1.0, poly.n_terms),
    ]
    with torch.no_grad():
        for m in range(2):
            poly.coefficients[m].copy_(coeffs[m])

    x = QUERY_X
    e = torch.tensor([400.0, 700.0, 900.0])
    got = poly.evaluate(torch.stack([x, e], dim=1)).reshape(-1)
    expected = brute_force_evaluate(poly, axes, exps, coeffs, x, e)
    assert torch.allclose(got, expected, atol=1e-5)


def test_assemble_matches_brute_force_with_nonzero_coefficients():
    axes = make_two_axes()
    exps = uniform_exponents(2, 3)
    poly = PolynomialNLPGD(axes=axes, n_modes_max=2, exponents=exps, n_modes_ini=2)
    seed_monoms(poly, 2)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.tensor([0.7, -0.3]))
        poly.coefficients[1].copy_(torch.tensor([-1.2, 0.4]))

    got = poly.assemble([QUERY_X, QUERY_E])
    assert got.shape == (3, 2)

    expected = torch.zeros(3, 2)
    for m in range(2):
        s = interpolate(poly, axes, m, 0, QUERY_X)
        g = interpolate(poly, axes, m, 1, QUERY_E)
        expected = expected + torch.outer(s, g)
        for t, lam in enumerate(exps):
            expected = expected + poly.coefficients[m][t] * torch.outer(
                s ** int(lam[0]), g ** int(lam[1])
            )
    assert torch.allclose(got, expected, atol=1e-5)


# --------------------------------------------------------------------------
# 4. evaluate vs assemble
# --------------------------------------------------------------------------


def test_evaluate_agrees_with_assemble_on_the_grid_diagonal():
    axes = make_two_axes()
    poly = PolynomialNLPGD(
        axes=axes, n_modes_max=2, exponents=total_degree_exponents(2, 4), n_modes_ini=2
    )
    seed_monoms(poly, 2)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.full((poly.n_terms,), 0.6))
        poly.coefficients[1].copy_(torch.full((poly.n_terms,), -0.25))

    x = torch.tensor([2.5, 5.0])
    e = torch.tensor([400.0, 700.0])
    grid = poly.assemble([x, e])
    diag = poly.evaluate(torch.stack([x, e], dim=1)).reshape(-1)
    assert torch.allclose(torch.diagonal(grid), diag, atol=1e-5)


# --------------------------------------------------------------------------
# 5. Lifecycle
# --------------------------------------------------------------------------


def test_construction_leaves_every_coefficient_row_frozen_at_zero():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=3,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    assert len(poly.coefficients) == 3
    for c in poly.coefficients:
        assert c.shape == (2,)
        assert not c.requires_grad
        assert torch.equal(c, torch.zeros(2))
    # ... while the initially-active modes' monoms are trainable.
    assert all(f.values_reduced.requires_grad for f in poly.monoms[0])
    assert all(f.values_reduced.requires_grad for f in poly.monoms[1])
    assert all(not f.values_reduced.requires_grad for f in poly.monoms[2])


def test_add_mode_does_not_release_coefficients():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    before = len(poly.polynomial_directory())
    m = poly.add_mode()

    assert m == 1
    assert int(poly.n_modes_truncated) == 2
    # The new mode's monoms are trainable, its coefficients are not.
    assert all(f.values_reduced.requires_grad for f in poly.monoms[1])
    assert not poly.coefficients[1].requires_grad
    # polynomial_directory grows by the whole mode: leading + |I|.
    assert len(poly.polynomial_directory()) == before + 1 + poly.n_terms
    # Trainable parameters gained only the two monoms.
    trainable = [p for p in poly.parameters() if p.requires_grad]
    assert len(trainable) == 4


def test_unfreeze_mode_coefficients_touches_only_that_row():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=3,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=3,
    )
    poly.unfreeze_mode_coefficients(1)

    assert not poly.coefficients[0].requires_grad
    assert poly.coefficients[1].requires_grad
    assert not poly.coefficients[2].requires_grad
    # Monoms untouched by the coefficient call.
    assert all(f.values_reduced.requires_grad for f in poly.monoms[1])

    poly.freeze_mode_coefficients(1)
    assert not poly.coefficients[1].requires_grad


def test_freeze_mode_is_monoms_only_and_freeze_all_covers_both():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    poly.unfreeze_mode_coefficients(0)
    poly.freeze_mode(0)

    # freeze_mode is inherited and monom-only: the coefficient row survives.
    assert all(not f.values_reduced.requires_grad for f in poly.monoms[0])
    assert poly.coefficients[0].requires_grad

    poly.freeze_all()
    assert all(not c.requires_grad for c in poly.coefficients)
    assert all(not f.values_reduced.requires_grad for mode in poly.monoms for f in mode)


def test_mode_parameters_include_the_coefficient_row():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    params = poly.mode_parameters(1)
    assert len(params) == 3  # 2 monoms + 1 coefficient row
    assert params[-1] is poly.coefficients[1]
    assert poly.mode_parameters()[-1] is poly.coefficients[1]  # default = last

    only_c = poly.mode_coefficient_parameters(0)
    assert only_c == [poly.coefficients[0]]

    with pytest.raises(IndexError):
        poly.mode_parameters(5)


def test_two_stage_protocol_end_to_end():
    """Stage 1 trains monoms with C pinned at 0; stage 2 trains the correction."""
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    seed_monoms(poly, 1)

    def loss():
        return separable_energy(
            fill_layout(poly), poly.directory(), poly.polynomial_directory()
        )

    # Stage 1: monoms only, C pinned at 0.
    monoms_before = [f.values_reduced.detach().clone() for f in poly.monoms[0]]
    opt = torch.optim.SGD([p for p in poly.parameters() if p.requires_grad], lr=1e-8)
    for _ in range(5):
        opt.zero_grad()
        # retain_graph as in PGDTrainer._closure: the quadrature contexts are
        # rebuilt per call but share saved tensors.
        loss().backward(retain_graph=True)
        opt.step()

    assert torch.equal(poly.coefficients[0], torch.zeros(poly.n_terms))
    assert any(
        not torch.equal(f.values_reduced, b)
        for f, b in zip(poly.monoms[0], monoms_before)
    )

    # Stage 2: the correction only.
    poly.freeze_mode(0)
    poly.unfreeze_mode_coefficients(0)
    opt = torch.optim.SGD(poly.mode_coefficient_parameters(0), lr=1e-8)
    monoms_before = [f.values_reduced.detach().clone() for f in poly.monoms[0]]
    for _ in range(5):
        opt.zero_grad()
        loss().backward(retain_graph=True)
        opt.step()

    assert float(poly.coefficients[0].detach().abs().max()) > 0.0
    assert all(
        torch.equal(f.values_reduced, b) for f, b in zip(poly.monoms[0], monoms_before)
    )


def test_polynomial_directory_layout():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.tensor([0.5, 0.25]))

    entries = poly.polynomial_directory()
    assert len(entries) == 2 * (1 + 2)

    modes = [e[0] for e in entries]
    assert modes == [0, 0, 0, 1, 1, 1]  # mode-major

    # Leading term first in each block: exponents all 1, coefficient None.
    assert entries[0][1] == (1, 1) and entries[0][2] is None
    assert entries[3][1] == (1, 1) and entries[3][2] is None

    # Corrections carry the exponent rows and the live coefficient elements.
    assert entries[1][1] == (2, 2)
    assert entries[2][1] == (3, 3)
    assert float(entries[1][2]) == pytest.approx(0.5)
    assert float(entries[2][2]) == pytest.approx(0.25)

    # Truncated to active modes, like directory().
    poly2 = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=3,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    assert len(poly2.polynomial_directory()) == 3
    assert len(poly2.directory()["space"]) == 1


# --------------------------------------------------------------------------
# 6. Guards
# --------------------------------------------------------------------------


def test_vector_valued_axis_is_rejected():
    axes = [make_axis(name="space", n=5, dim=2), make_axis(name="E", n=4)]
    with pytest.raises(ValueError, match="requires scalar axes"):
        PolynomialNLPGD(axes=axes, n_modes_max=1, exponents=uniform_exponents(2, 3))


@pytest.mark.parametrize(
    "exponents, match",
    [
        (torch.tensor([[2, 2, 2]]), r"n_axes = 2"),
        (torch.tensor([2, 2]), r"n_axes = 2"),
        (torch.tensor([[2.0, 2.0]]), "integer dtype"),
        (torch.tensor([[0, 2]]), "must be >= 1"),
        (torch.tensor([[1, 1], [2, 2]]), r"must not contain \(1, \.\.\., 1\)"),
        (torch.tensor([[2, 2], [2, 2]]), "duplicate rows"),
        (torch.zeros(0, 2, dtype=torch.long), "is empty"),
    ],
)
def test_malformed_exponents_are_rejected(exponents, match):
    with pytest.raises(ValueError, match=match):
        PolynomialNLPGD(axes=make_two_axes(), n_modes_max=1, exponents=exponents)


# --------------------------------------------------------------------------
# 7. Registration and state round-trip
# --------------------------------------------------------------------------


def test_register_into_registers_every_monom():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    layout = FieldLayout()
    poly.register_into(layout)
    for mode in poly.monoms:
        for f in mode:
            assert layout._fields[f.name] is f


def test_state_dict_round_trip_preserves_coefficients_and_exponents():
    exps = total_degree_exponents(2, 4)
    poly = PolynomialNLPGD(
        axes=make_two_axes(), n_modes_max=2, exponents=exps, n_modes_ini=2
    )
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.linspace(0.1, 0.9, poly.n_terms))
        poly.coefficients[1].copy_(torch.linspace(-0.9, -0.1, poly.n_terms))
    state = poly.state_dict()
    assert "exponents" in state

    restored = PolynomialNLPGD(
        axes=make_two_axes(), n_modes_max=2, exponents=exps, n_modes_ini=2
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.exponents, poly.exponents)
    for a, b in zip(restored.coefficients, poly.coefficients):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# 8. Gauge diagnostics (reported, not asserted)
# --------------------------------------------------------------------------


def apply_gauge(poly, scales):
    """Apply ``w_j -> s_j w_j``, ``C_lambda -> C_lambda prod_j s_j^(-lambda_j)``."""
    with torch.no_grad():
        for k, s in enumerate(scales):
            poly.monoms[0][k].values_reduced.mul_(s)
        for t in range(poly.n_terms):
            factor = 1.0
            for k, s in enumerate(scales):
                factor *= s ** (-int(poly.exponents[t, k]))
            poly.coefficients[0][t] *= factor


def build_seeded(exponents):
    poly = PolynomialNLPGD(
        axes=make_two_axes(), n_modes_max=1, exponents=exponents, n_modes_ini=1
    )
    seed_monoms(poly, 1)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.linspace(0.3, 0.8, poly.n_terms))
    return poly


def energy_of(poly):
    return float(
        separable_energy(
            fill_layout(poly), poly.directory(), poly.polynomial_directory()
        ).detach()
    )


@pytest.mark.parametrize(
    "exponents", [uniform_exponents(2, 3), total_degree_exponents(2, 4)]
)
def test_energy_is_invariant_along_the_gauge_orbit(exponents):
    """The degeneracy is exact: prod_j s_j = 1 leaves the energy untouched.

    Documents *which* transformation is the symmetry. The `prod_j s_j = 1`
    constraint comes from the leading term's coefficient being pinned at 1, so
    the orbit has d - 1 dimensions per mode, not d.
    """
    poly = build_seeded(exponents)
    before = energy_of(poly)
    s = 1.7
    apply_gauge(poly, [s, 1.0 / s])  # prod = 1
    assert energy_of(poly) == pytest.approx(before, rel=1e-6)


def test_energy_is_not_invariant_when_the_scales_do_not_multiply_to_one():
    """The counterpart: free rescaling is *not* a symmetry.

    Guards against restating the degeneracy as `w -> s w, C -> C s^{-sum lambda}`,
    which is a symmetry of one term in isolation but not of the mode -- the
    leading term is not scale-free.
    """
    poly = build_seeded(uniform_exponents(2, 3))
    before = energy_of(poly)
    apply_gauge(poly, [1.7, 1.7])  # prod = 2.89 != 1
    assert energy_of(poly) != pytest.approx(before, rel=1e-3)
