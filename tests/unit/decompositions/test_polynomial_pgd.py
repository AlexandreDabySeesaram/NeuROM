import pytest
import torch

from neurom.decompositions import (
    Axis,
    CPPGD,
    LegendreBasis,
    MonomialBasis,
    PolynomialNLPGD,
    pin_axis,
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


def test_term_is_inert_needs_both_frozen_and_zero():
    """The full truth table. Frozen-and-zero only; every other cell is live.

    The `zero but trainable` cell is the one that matters: it is the ordinary
    state of a fresh mode at the start of a `joint` stage, and calling it inert
    would drop it from the graph, starve it of gradient and freeze it at zero
    for good -- a correction that silently never activates.
    """
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.tensor([0.0, 0.25]))

    # Frozen (construction default): zero is inert, non-zero is not.
    assert poly.coefficients[0].requires_grad is False
    assert poly.term_is_inert(0, 0) is True
    assert poly.term_is_inert(0, 1) is False

    # Trainable: neither is inert, the zero one least of all.
    poly.unfreeze_mode_coefficients(0)
    assert poly.term_is_inert(0, 0) is False
    assert poly.term_is_inert(0, 1) is False


def test_skip_inert_drops_only_the_frozen_zero_terms():
    """Counts, per mode, against a mixed decomposition.

    Mode 0 frozen with one zero and one non-zero coefficient -- the state
    ``staged`` leaves behind -- and mode 1 trainable at zero, the state a fresh
    mode starts in.
    """
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.tensor([0.0, 0.25]))
    poly.unfreeze_mode_coefficients(1)

    assert len(poly.polynomial_directory()) == 2 * (1 + 2)
    entries = poly.polynomial_directory(skip_inert=True)

    # Mode 0 loses its zero frozen term; mode 1 keeps both, being trainable.
    assert [e[0] for e in entries] == [0, 0, 1, 1, 1]
    assert [e[1] for e in entries] == [(1, 1), (3, 3), (1, 1), (2, 2), (3, 3)]

    # Leading terms are never skipped, whatever the coefficients do.
    assert sum(1 for e in entries if e[1] == (1, 1)) == 2


def test_skip_inert_leaves_the_energy_unchanged():
    """Same number, fewer terms. The point is cost, never the field."""
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    seed_monoms(poly, 2)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.tensor([0.0, 0.25]))
        poly.coefficients[1].copy_(torch.tensor([0.0, 0.0]))

    layout, directory = fill_layout(poly), poly.directory()
    full = separable_energy(layout, directory, poly.polynomial_directory())
    lean = separable_energy(
        layout, directory, poly.polynomial_directory(skip_inert=True)
    )
    assert float(lean) == pytest.approx(float(full), rel=1e-12)
    # And it did actually drop something, or the equality above proves nothing.
    # 6 -> 3: mode 0 keeps its leading term and its non-zero correction, mode 1
    # is frozen at zero throughout so it is down to its leading term alone.
    assert len(poly.polynomial_directory()) == 6
    assert len(poly.polynomial_directory(skip_inert=True)) == 3


def test_skip_inert_keeps_the_gradient_of_a_trainable_zero_coefficient():
    """The guard that makes `skip_inert` safe, checked on the real gradient path.

    A fresh mode's ``C`` is zero *and* trainable. Under a naive ``C == 0`` test
    it would be skipped, get no gradient, and never leave zero. Here it must
    receive the same gradient it does without ``skip_inert``.
    """
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )
    seed_monoms(poly, 1)
    poly.unfreeze_mode_coefficients(0)
    assert float(poly.coefficients[0].abs().max()) == 0.0

    # `retain_graph`: the two energies share the interpolation graph under the
    # layout, so the first backward would free what the second needs.
    separable_energy(
        fill_layout(poly), poly.directory(),
        poly.polynomial_directory(skip_inert=True),
    ).backward(retain_graph=True)
    lean_grad = poly.coefficients[0].grad.clone()
    assert float(lean_grad.abs().min()) > 0.0, "a skipped term would show 0 here"

    poly.coefficients[0].grad = None
    separable_energy(
        fill_layout(poly), poly.directory(), poly.polynomial_directory()
    ).backward()
    assert torch.allclose(lean_grad, poly.coefficients[0].grad, atol=1e-12)


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


# --------------------------------------------------------------------------
# 9. Gauge fixing (renormalise)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exponents", [uniform_exponents(2, 3), total_degree_exponents(2, 4)]
)
def test_renormalise_preserves_the_field(exponents):
    """The gauge fix is an exact reparameterisation, not a regularisation."""
    poly = build_seeded(exponents)
    before_energy = energy_of(poly)
    before_grid = poly.assemble([QUERY_X, QUERY_E])

    poly.renormalise()

    assert energy_of(poly) == pytest.approx(before_energy, rel=1e-5)
    assert torch.allclose(poly.assemble([QUERY_X, QUERY_E]), before_grid, atol=1e-4)


def test_renormalise_puts_unit_norm_on_the_last_axes():
    """The last d - 1 monoms end at unit quadrature norm; axis 0 takes the scale."""
    poly = build_seeded(uniform_exponents(2, 3))
    before = [float(n) for n in poly.monom_norms(0)]
    poly.renormalise()

    norms = [float(n) for n in poly.monom_norms(0)]
    for k in range(1, len(norms)):
        assert norms[k] == pytest.approx(1.0, rel=1e-5)
    # Axis 0 absorbed the scale: ||w_0|| ends at the product of all the
    # original norms (s_0 = prod_{j>=1} ||w_j||, applied to ||w_0||).
    expected = 1.0
    for n in before:
        expected *= n
    assert norms[0] == pytest.approx(expected, rel=1e-5)


def test_renormalise_is_idempotent():
    poly = build_seeded(total_degree_exponents(2, 4))
    poly.renormalise()
    once = [f.values_reduced.detach().clone() for f in poly.monoms[0]]
    once_c = poly.coefficients[0].detach().clone()

    poly.renormalise()

    for f, b in zip(poly.monoms[0], once):
        assert torch.allclose(f.values_reduced, b, atol=1e-6)
    assert torch.allclose(poly.coefficients[0], once_c, atol=1e-6)


def test_renormalise_covers_every_active_mode_including_frozen_ones():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=3,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=2,
    )
    seed_monoms(poly, 2)
    poly.freeze_mode(0)
    before = poly.assemble([QUERY_X, QUERY_E])

    poly.renormalise()

    for m in range(2):
        assert float(poly.monom_norms(m)[1]) == pytest.approx(1.0, rel=1e-5)
    # Frozen mode 0 was rescaled but its contribution to the field is unchanged.
    assert torch.allclose(poly.assemble([QUERY_X, QUERY_E]), before, atol=1e-4)


def test_renormalise_skips_a_mode_with_a_zero_monom():
    """An unseeded monom has no scale to normalise; leave it alone, do not divide by 0."""
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=uniform_exponents(2, 3),
        n_modes_ini=1,
    )  # init_values are all zero
    poly.renormalise()
    for f in poly.monoms[0]:
        assert torch.all(torch.isfinite(f.values_reduced))
        assert torch.equal(f.values_reduced, torch.zeros_like(f.values_reduced))


def constrained_axes(imposed):
    from neurom.constraints import Dirichlet

    axes = make_two_axes()
    axes[0] = Axis(
        name="space",
        nodes_positions=axes[0].nodes_positions,
        sf=axes[0].sf,
        mapping=axes[0].mapping,
        quad=axes[0].quad,
        constraint=Dirichlet(
            nodes=torch.tensor([0]), values_imposed=torch.tensor([[imposed]])
        ),
        init_values=torch.zeros(5, 1),
    )
    return axes


def test_renormalise_rejects_non_homogeneous_constraints():
    """Construction is fine; only the gauge fix itself is illegal."""
    poly = PolynomialNLPGD(
        axes=constrained_axes(2.0), n_modes_max=1, exponents=uniform_exponents(2, 3)
    )
    with pytest.raises(ValueError, match="homogeneous constraints"):
        poly.renormalise()


def test_renormalise_accepts_homogeneous_dirichlet():
    poly = PolynomialNLPGD(
        axes=constrained_axes(0.0), n_modes_max=1, exponents=uniform_exponents(2, 3)
    )
    poly.renormalise()  # must not raise


def test_cppgd_renormalise_is_a_no_op():
    """The trainer calls renormalise() unconditionally, so CPPGD must accept it."""
    cp = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    seed_monoms(cp, 1)
    before = [f.values_reduced.detach().clone() for f in cp.monoms[0]]
    cp.renormalise()
    for f, b in zip(cp.monoms[0], before):
        assert torch.equal(f.values_reduced, b)


def test_truncated_drops_the_trailing_mode_including_its_coefficients():
    """Truncation must remove a polynomial mode whole -- monoms *and* its ``C``.

    Worth its own test beyond the CPPGD one: a polynomial mode carries a second
    parameter family, and a truncation that dropped only the monoms would leave
    the coefficient row contributing and put a wrong point on the rank curve.
    """
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

    query = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 900.0])], dim=1)
    full = poly.evaluate(query).reshape(-1)

    with poly.truncated(1):
        got = poly.evaluate(query).reshape(-1)
        expected = brute_force_evaluate(
            poly, axes, exps, coeffs[:1], QUERY_X,
            torch.tensor([400.0, 700.0, 900.0]),
        )
        assert torch.allclose(got, expected, atol=1e-5)
        assert not torch.allclose(got, full)

    assert torch.allclose(poly.evaluate(query).reshape(-1), full)


# --------------------------------------------------------------------------
# 11. pin_axis
# --------------------------------------------------------------------------


def test_pin_axis_forces_the_column_and_leaves_the_rest():
    pinned = pin_axis(uniform_exponents(5, 3), axis=0, power=1)
    assert [tuple(int(v) for v in r) for r in pinned] == [
        (1, 2, 2, 2, 2),
        (1, 3, 3, 3, 3),
    ]


def test_pin_axis_deduplicates_and_drops_the_leading_term():
    """total_degree collapses onto itself once a column is pinned."""
    full = total_degree_exponents(3, 5)
    pinned = pin_axis(full, axis=0, power=1)
    rows = [tuple(int(v) for v in r) for r in pinned]

    assert len(rows) == len(set(rows)), "rows must be unique"
    assert (1, 1, 1) not in rows, "the leading term is carried separately"
    assert all(r[0] == 1 for r in rows)
    assert len(rows) < len(full), "this set really does collapse"
    # (2,1,1) pins to (1,1,1) and must be dropped, not kept as a duplicate
    # leading term -- that is the case that would silently double the mode.
    assert (2, 1, 1) in [tuple(int(v) for v in r) for r in full]


def test_pin_axis_result_is_a_valid_exponent_set():
    """Whatever it returns must survive PolynomialNLPGD's own validation."""
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=pin_axis(uniform_exponents(2, 4), axis=0, power=1),
    )
    assert poly.n_terms == 3  # (1,2), (1,3), (1,4)


def test_pin_axis_accepts_a_negative_axis():
    pinned = pin_axis(uniform_exponents(3, 3), axis=-1, power=1)
    assert all(int(r[-1]) == 1 for r in pinned)


def test_pin_axis_rejects_a_zero_power():
    with pytest.raises(ValueError, match="power must be >= 1"):
        pin_axis(uniform_exponents(2, 3), axis=0, power=0)


def test_pin_axis_rejects_an_out_of_range_axis():
    with pytest.raises(ValueError, match="out of range"):
        pin_axis(uniform_exponents(2, 3), axis=5, power=1)


def test_pin_axis_rejects_a_set_that_collapses_to_nothing():
    """uniform(2, 3) = {(2,2), (3,3)}; pinning BOTH columns leaves only (1,1)."""
    with pytest.raises(ValueError, match="left the exponent set empty"):
        pin_axis(pin_axis(uniform_exponents(2, 3), 0, 1), 1, 1)


# --------------------------------------------------------------------------
# 12. Optional leading coefficient
# --------------------------------------------------------------------------


def build_seeded_leading(exponents=None, n_modes_max=1):
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=n_modes_max,
        exponents=exponents if exponents is not None else uniform_exponents(2, 3),
        n_modes_ini=1,
        leading_coefficients=True,
    )
    seed_monoms(poly, 1)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.linspace(0.3, 0.8, poly.n_terms))
    return poly


def test_leading_coefficients_off_by_default_and_add_no_state():
    """The flag must be free: no parameter, no state_dict key, no behaviour change."""
    plain = PolynomialNLPGD(
        axes=make_two_axes(), n_modes_max=2, exponents=uniform_exponents(2, 3)
    )
    assert plain.has_leading_coefficients is False
    assert plain.leading_coefficients is None
    assert not any("leading" in k for k in plain.state_dict())

    with pytest.raises(RuntimeError, match="no leading coefficients"):
        plain.unfreeze_mode_leading_coefficient(0)


def test_leading_coefficients_on_adds_exactly_the_expected_state():
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=2,
        exponents=uniform_exponents(2, 3),
        leading_coefficients=True,
    )
    keys = [k for k in poly.state_dict() if "leading_coefficients" in k]
    assert len(keys) == 2  # one scalar per mode
    assert all(float(c) == 1.0 for c in poly.leading_coefficients)
    assert all(not c.requires_grad for c in poly.leading_coefficients)


def test_a_unit_leading_coefficient_is_bitwise_the_fixed_weight():
    """c = 1 must reproduce the fixed-weight decomposition exactly, not nearly."""
    exponents = uniform_exponents(2, 3)
    plain = build_seeded(exponents)
    lead = build_seeded_leading(exponents)

    query = torch.stack(
        [QUERY_X.repeat_interleave(len(QUERY_E)), QUERY_E.repeat(len(QUERY_X))], dim=1
    )
    assert torch.equal(plain.evaluate(query), lead.evaluate(query))
    assert torch.equal(
        plain.assemble([QUERY_X, QUERY_E]), lead.assemble([QUERY_X, QUERY_E])
    )


def test_the_leading_coefficient_scales_only_the_leading_term():
    poly = build_seeded_leading()
    grid = poly.assemble([QUERY_X, QUERY_E])

    with torch.no_grad():
        poly.coefficients[0].zero_()  # corrections off: only the leading term left
    leading_only = poly.assemble([QUERY_X, QUERY_E])

    with torch.no_grad():
        poly.leading_coefficients[0].fill_(3.0)
    assert torch.allclose(
        poly.assemble([QUERY_X, QUERY_E]), 3.0 * leading_only, atol=1e-5
    )
    assert not torch.allclose(leading_only, grid)


def test_polynomial_directory_reports_the_leading_coefficient():
    poly = build_seeded_leading()
    terms = poly.polynomial_directory()

    mode, exponents, coefficient = terms[0]
    assert (mode, exponents) == (0, (1, 1))
    assert coefficient is poly.leading_coefficients[0]

    # ... and still None when the weight is fixed, so existing consumers of the
    # `1.0 if coefficient is None else coefficient` idiom are unaffected.
    assert build_seeded(uniform_exponents(2, 3)).polynomial_directory()[0][2] is None


def test_leading_coefficient_lifecycle():
    poly = build_seeded_leading()
    assert not poly.leading_coefficients[0].requires_grad

    poly.unfreeze_mode_leading_coefficient(0)
    assert poly.leading_coefficients[0].requires_grad
    assert poly.mode_coefficient_parameters(0) == [
        poly.leading_coefficients[0],
        poly.coefficients[0],
    ]
    assert poly.mode_parameters(0)[-2] is poly.leading_coefficients[0]

    poly.freeze_all()
    assert not poly.leading_coefficients[0].requires_grad


@pytest.mark.parametrize(
    "exponents", [uniform_exponents(2, 3), total_degree_exponents(2, 4)]
)
def test_renormalise_with_a_leading_coefficient_preserves_the_field(exponents):
    poly = build_seeded_leading(exponents)
    with torch.no_grad():
        poly.leading_coefficients[0].fill_(2.5)  # not 1, so the c-path is exercised
    before_energy = energy_of(poly)
    before_grid = poly.assemble([QUERY_X, QUERY_E])

    poly.renormalise()

    assert energy_of(poly) == pytest.approx(before_energy, rel=1e-5)
    assert torch.allclose(poly.assemble([QUERY_X, QUERY_E]), before_grid, atol=1e-4)


def test_renormalise_with_a_leading_coefficient_normalises_every_axis():
    """d conditions, not d - 1: the amplitude moves onto c, not onto axis 0."""
    poly = build_seeded_leading()
    before = [float(n) for n in poly.monom_norms(0)]
    assert not all(n == pytest.approx(1.0, rel=1e-4) for n in before)

    poly.renormalise()

    after = [float(n) for n in poly.monom_norms(0)]
    assert all(n == pytest.approx(1.0, rel=1e-4) for n in after)
    # The scale went to c, which started at 1.
    assert float(poly.leading_coefficients[0]) == pytest.approx(
        before[0] * before[1], rel=1e-4
    )


def test_renormalise_with_a_leading_coefficient_puts_every_term_on_one_scale():
    """The claim the whole option exists for.

    With every monom at unit norm, a term's natural size
    ``prod_j ||w_ij||^lambda_j`` is 1 for EVERY exponent row -- so the leading
    coefficient and every row of C share a scale. Under the d-1 fix the same
    quantity is ``A^p``, which is what forces one Adam ``coefficient_lr`` to
    serve rows orders of magnitude apart.
    """
    poly = build_seeded_leading(uniform_exponents(2, 4))
    poly.renormalise()
    norms = poly.monom_norms(0)

    for row in poly.exponents:
        natural_size = 1.0
        for k, power in enumerate(row):
            natural_size *= float(norms[k]) ** int(power)
        assert natural_size == pytest.approx(1.0, rel=1e-3)

    # ... and the contrast: without the leading coefficient, axis 0 keeps the
    # amplitude A and the p-th row's natural size is A^p, not 1.
    plain = build_seeded(uniform_exponents(2, 4))
    plain.renormalise()
    plain_norms = plain.monom_norms(0)
    amplitude = float(plain_norms[0])
    assert amplitude != pytest.approx(1.0, rel=1e-2)
    for row in plain.exponents:
        natural_size = 1.0
        for k, power in enumerate(row):
            natural_size *= float(plain_norms[k]) ** int(power)
        assert natural_size == pytest.approx(amplitude ** int(row[0]), rel=1e-3)


def test_renormalise_with_a_leading_coefficient_is_idempotent():
    poly = build_seeded_leading(total_degree_exponents(2, 4))
    poly.renormalise()
    once = [f.values_reduced.detach().clone() for f in poly.monoms[0]]
    once_c = poly.leading_coefficients[0].detach().clone()

    poly.renormalise()

    assert all(
        torch.allclose(f.values_reduced, b, atol=1e-6)
        for f, b in zip(poly.monoms[0], once)
    )
    assert torch.allclose(poly.leading_coefficients[0], once_c, atol=1e-6)


# --------------------------------------------------------------------------
# 8. Orthogonal corrections
# --------------------------------------------------------------------------


def build_orthogonal(exponents, n_modes=1, leading=False):
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=n_modes,
        exponents=exponents,
        n_modes_ini=n_modes,
        leading_coefficients=leading,
        orthogonal_corrections=True,
    )
    seed_monoms(poly, n_modes)
    with torch.no_grad():
        for m in range(n_modes):
            poly.coefficients[m].copy_(torch.linspace(0.3, 0.8, poly.n_terms))
    return poly


def test_orthogonal_corrections_off_changes_nothing():
    # The guard on every existing checkpoint: the flag adds no parameter, so the
    # state_dict must be identical, and so must the field.
    exponents = uniform_exponents(2, 3)
    plain = build_seeded(exponents)
    flagged = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=exponents,
        n_modes_ini=1,
        orthogonal_corrections=True,
    )

    assert set(plain.state_dict()) == set(flagged.state_dict())

    coords = torch.stack(
        [QUERY_X, torch.tensor([400.0, 700.0, 1000.0])], dim=1
    )
    on = build_orthogonal(exponents)
    assert not torch.allclose(plain.evaluate(coords), on.evaluate(coords))


def test_the_deflation_axis_is_the_first_non_unit_power():
    # lambda_j = 1 would deflate the factor to exactly zero and annihilate the
    # whole term, so the axis must be one whose power is not 1.
    assert PolynomialNLPGD._deflation_axis((1, 2, 1)) == 1
    assert PolynomialNLPGD._deflation_axis((3, 3)) == 0
    assert PolynomialNLPGD._deflation_axis((1, 1, 4)) == 2
    with pytest.raises(ValueError, match="leading term"):
        PolynomialNLPGD._deflation_axis((1, 1))


def leading_inner_products(poly, m, layout):
    """``int w_ij^p w_ij dmu_j`` per axis, for every power the rows use."""
    directory = poly.directory()
    axis_names = list(directory)

    def moment(k, power):
        r = layout[directory[axis_names[k]][m]]
        return float(integrate(r.u**power * r.u * r.measure))

    return moment


def test_a_deflated_correction_is_orthogonal_to_the_leading_term():
    # The claim the whole option rests on. <prod_j f_j, prod_j w_j> factorises
    # into per-axis integrals, so the correction's overlap with the leading term
    # is the sum, over the two rows it expands into, of the product of those
    # integrals -- and it must vanish.
    poly = build_orthogonal(uniform_exponents(2, 3))
    layout = fill_layout(poly)
    moment = leading_inner_products(poly, 0, layout)

    rows = poly._term_rows(0)
    leading, corrections = rows[0], rows[1:]
    assert leading[0] == (1, 1)

    # Rows come in (lam, C), (shadow, -beta C) pairs, one pair per correction.
    assert len(corrections) == 2 * poly.n_terms
    for i in range(poly.n_terms):
        overlap = 0.0
        for lam, coefficient in corrections[2 * i : 2 * i + 2]:
            contribution = float(coefficient)
            for k, power in enumerate(lam):
                contribution *= moment(k, power)
            overlap += contribution
        # Scale-free: compare against the size of either row on its own, or a
        # cancellation between two large numbers would pass trivially.
        alone = abs(float(corrections[2 * i][1]))
        for k, power in enumerate(corrections[2 * i][0]):
            alone *= abs(moment(k, power))
        assert abs(overlap) < 1e-5 * alone


def test_the_undeflated_correction_is_not_orthogonal():
    # The contrast, so the test above cannot pass by the overlap being zero for
    # some unrelated reason (an odd monom, say).
    poly = build_seeded(uniform_exponents(2, 3))
    layout = fill_layout(poly)
    moment = leading_inner_products(poly, 0, layout)

    lam, coefficient = poly._term_rows(0)[1]
    overlap = float(coefficient)
    for k, power in enumerate(lam):
        overlap *= moment(k, power)

    assert abs(overlap) > 1e-3


def test_evaluate_and_the_directory_describe_the_same_field():
    # The consistency that matters most: `energy` reads `polynomial_directory`
    # while `evaluate`/`assemble` read `_mode_from_columns`. If deflation reached
    # only one of them, a run would train one field and report another.
    poly = build_orthogonal(uniform_exponents(2, 3), n_modes=1)
    axes = make_two_axes()
    coords = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 1000.0])], dim=1)

    cols = [interpolate(poly, axes, 0, k, coords[:, k]) for k in range(2)]
    expected = torch.zeros(coords.shape[0])
    for mode, lam, coefficient in poly.polynomial_directory():
        term = torch.ones(coords.shape[0])
        for k, power in enumerate(lam):
            term = term * cols[k] ** power
        weight = 1.0 if coefficient is None else float(coefficient)
        expected = expected + weight * term

    assert torch.allclose(poly.evaluate(coords).reshape(-1), expected, atol=1e-5)


def test_deflation_is_inert_while_the_coefficients_are_zero():
    # A decomposition whose C rows are still at 0 must be exactly CPPGD, flag or
    # no flag -- otherwise every run's linear phase would already be perturbed.
    exponents = uniform_exponents(2, 3)
    coords = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 1000.0])], dim=1)

    poly = build_orthogonal(exponents)
    with torch.no_grad():
        poly.coefficients[0].zero_()

    cp = CPPGD(axes=make_two_axes(), n_modes_max=1, n_modes_ini=1)
    seed_monoms(cp, 1)

    assert torch.allclose(poly.evaluate(coords), cp.evaluate(coords), atol=1e-6)


def test_deflation_survives_skip_inert():
    # Both rows of a deflated correction must disappear together: leaving the
    # shadow behind would evaluate `-beta C prod w` with no term to cancel it.
    poly = build_orthogonal(uniform_exponents(2, 3))
    poly.freeze_mode_coefficients(0)
    with torch.no_grad():
        poly.coefficients[0].zero_()

    assert poly._term_rows(0, skip_inert=True) == [((1, 1), None)]


def test_the_deflation_factor_carries_a_gradient():
    # `monom_norms` runs under no_grad because the gauge fix is applied to the
    # parameters; beta is composed into the field instead, so a gradient that
    # ignored it would be the gradient of a different model.
    poly = build_orthogonal(uniform_exponents(2, 2))
    fill_layout(poly)
    beta = poly.deflation_factor(0, 0, 2)

    assert beta.requires_grad


# --------------------------------------------------------------------------
# 9. Injected term bases
# --------------------------------------------------------------------------


def build_with_bases(exponents, bases, leading=False):
    poly = PolynomialNLPGD(
        axes=make_two_axes(),
        n_modes_max=1,
        exponents=exponents,
        n_modes_ini=1,
        leading_coefficients=leading,
        bases=bases,
    )
    seed_monoms(poly, 1)
    with torch.no_grad():
        poly.coefficients[0].copy_(torch.linspace(0.3, 0.8, poly.n_terms))
    return poly


def legendre_on_the_second_axis():
    """Monomial on axis 0, Legendre on axis 1 -- the beam's arrangement in small.

    Axis 0 stands for `space`, which must keep the monomials because its
    Dirichlet condition makes the monom vanish at both ends and `P_2(0) != 0`.
    """
    return [MonomialBasis(), LegendreBasis()]


def test_no_bases_is_the_monomial_family_and_changes_nothing():
    # The guard on every existing checkpoint and every number in the ledger.
    exponents = uniform_exponents(2, 3)
    coords = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 1000.0])], dim=1)

    implicit = build_seeded(exponents)
    explicit = build_with_bases(exponents, [MonomialBasis(), MonomialBasis()])

    assert set(implicit.state_dict()) == set(explicit.state_dict())
    assert torch.equal(implicit.evaluate(coords), explicit.evaluate(coords))
    assert implicit.has_uniform_monomial_basis


def test_bases_must_have_one_entry_per_axis():
    with pytest.raises(ValueError, match="one family per axis"):
        PolynomialNLPGD(
            axes=make_two_axes(),
            n_modes_max=1,
            exponents=uniform_exponents(2, 2),
            bases=[MonomialBasis()],
        )


def test_orthogonal_corrections_and_a_non_monomial_basis_are_exclusive():
    # Both attack the leading-vs-correction overlap; deflating an already
    # orthogonal family subtracts ~0 and only doubles the term count.
    with pytest.raises(ValueError, match="same overlap"):
        PolynomialNLPGD(
            axes=make_two_axes(),
            n_modes_max=1,
            exponents=uniform_exponents(2, 2),
            orthogonal_corrections=True,
            bases=legendre_on_the_second_axis(),
        )


def test_the_normalised_argument_stays_inside_the_unit_interval():
    # The whole point of the sup norm rather than the L2 one: Legendre is only
    # orthogonal on [-1, 1], and an L2 normalisation leaves the peaks outside.
    poly = build_with_bases(uniform_exponents(2, 2), legendre_on_the_second_axis())
    fill_layout(poly)
    r = poly._assemblies[0][1].interpolate()
    scaled = r.u / poly.monom_scale(0, 1)

    assert float(scaled.abs().max()) == pytest.approx(1.0)
    # ... and the monomial axis is left alone, at a plain 1.0.
    assert poly.monom_scale(0, 0) == 1.0


def test_basis_derivative_carries_the_normalising_scale():
    # The silent-failure test. On the monomial path scale == 1, so a dropped or
    # doubled 1/scale is invisible; under Legendre it is a quietly wrong energy
    # gradient. Checked against autograd through the whole seam.
    poly = build_with_bases(uniform_exponents(2, 2), legendre_on_the_second_axis())
    fill_layout(poly)
    values = torch.tensor([-0.4, 0.1, 0.6], requires_grad=True)

    poly.basis_value(0, 1, 2, values).sum().backward()

    assert torch.allclose(poly.basis_derivative(0, 1, 2, values.detach()), values.grad)


def test_a_legendre_factor_is_invariant_when_its_monom_is_rescaled():
    # psi_p(w / ||w||_inf) does not move with w's scale -- for the corrections
    # AND for the leading term, since psi_1 is the identity of the same
    # normalised argument. That is why renormalise_mode must skip such an axis.
    poly = build_with_bases(uniform_exponents(2, 2), legendre_on_the_second_axis())
    fill_layout(poly)
    before = poly.basis_value(0, 1, 2, poly._assemblies[0][1].interpolate().u).clone()

    with torch.no_grad():
        poly.monoms[0][1].values_reduced.mul_(3.0)
    fill_layout(poly)
    after = poly.basis_value(0, 1, 2, poly._assemblies[0][1].interpolate().u)

    assert torch.allclose(before, after, atol=1e-10)


def test_renormalise_is_field_preserving_with_a_legendre_axis():
    # renormalise_mode acts on the homogeneous axes only. If it applied the
    # s^(-lambda) rule to the Legendre axis too it would change the field, which
    # is exactly what this compares.
    poly = build_with_bases(
        uniform_exponents(2, 2), legendre_on_the_second_axis(), leading=True
    )
    with torch.no_grad():
        poly.leading_coefficients[0].fill_(1.3)
    grid = [QUERY_X, torch.tensor([400.0, 700.0])]
    before = poly.assemble(grid).clone()
    coefficient_before = poly.coefficients[0].detach().clone()

    poly.renormalise()

    assert torch.allclose(poly.assemble(grid), before, atol=1e-6)
    # The Legendre axis contributes no factor, so C moves only by the monomial
    # axis' scale -- and here axis 0 is the only gauged one.
    assert not torch.allclose(poly.coefficients[0], coefficient_before)


def test_factor_norm_is_the_norm_of_the_factor_not_the_powered_norm():
    # The bug this replaced: ||w^p|| != ||w||^p, and the two agree only at p = 1.
    poly = build_seeded(uniform_exponents(2, 2))
    fill_layout(poly)

    for k in range(2):
        assert float(poly.factor_norm(0, k, 1)) == pytest.approx(
            float(poly.monom_norms(0)[k]), rel=1e-9
        )

    powered = float(poly.monom_norms(0)[0]) ** 2
    assert float(poly.factor_norm(0, 0, 2)) != pytest.approx(powered, rel=1e-3)


def test_legendre_leaves_less_overlap_with_the_leading_term_than_the_monomials():
    # THE measurement the basis choice rests on. Legendre is orthogonal for the
    # uniform measure, but <psi_a(w), psi_b(w)> integrates against the law of
    # w's VALUES, which is not uniform -- so orthogonality here is approximate
    # and its quality is a number, not an assumption. Asserted as a ratio
    # against the monomial family rather than against a fixed threshold.
    def overlap(poly):
        # <prod_j psi_{lam_j}(w_j), prod_j w_j> factorises into per-axis
        # integrals; normalised by each factor's own norm so the result is a
        # cosine and cannot be made small by shrinking the term.
        fill_layout(poly)
        lam = [int(v) for v in poly.exponents[0]]
        cosine = 1.0
        for k in range(2):
            r = poly._assemblies[0][k].interpolate()
            f = poly.basis_value(0, k, lam[k], r.u)
            g = poly.basis_value(0, k, 1, r.u)
            num = float(integrate(f * g * r.measure))
            den = float(poly.factor_norm(0, k, lam[k])) * float(
                poly.factor_norm(0, k, 1)
            )
            cosine *= num / den
        return abs(cosine)

    exponents = uniform_exponents(2, 2)
    monomial = overlap(build_seeded(exponents))
    legendre = overlap(build_with_bases(exponents, [LegendreBasis(), LegendreBasis()]))

    # MEASURED on this fixture: monomial 0.949, Legendre 0.476 -- a factor 2, not
    # the orthogonality the family is named for. That is the expected shape of
    # the answer, not a failure: <psi_a(w), psi_b(w)> integrates against the law
    # of w's values, and Legendre is orthogonal for the *uniform* law. How far
    # the two laws are apart is exactly what this number reports.
    #
    # The bound is loose on purpose. It pins the direction (Legendre reduces the
    # overlap, and substantially) without pretending to a precision the fixture
    # cannot support: 4 and 5 nodes with hand-written monom values, so the value
    # distribution is whatever `MONOM_VALUES` happens to be. The number that
    # decides whether the basis is good enough is the one on the real problem.
    assert legendre < 0.6 * monomial


def test_degree_one_is_never_normalised():
    # THE property that keeps the linear phase working. psi_1 is the identity, so
    # w and w/||w||_inf differ by a positive constant and orthogonality against
    # the higher degrees is untouched -- but normalising degree 1 makes the
    # leading term scale-invariant, which quotients the mode's amplitude away and
    # leaves ONE multiplicative channel where CP has d. Measured on the beam,
    # rank 1, 300 iterations: stage-0 energy -6.64e10 against -1.90e11.
    assert not LegendreBasis().normalises(1)
    assert LegendreBasis().normalises(2)
    assert not MonomialBasis().normalises(2)


def test_a_frozen_correction_makes_a_legendre_decomposition_exactly_cp():
    # The consequence: with C still at zero -- every linear phase -- a Legendre
    # decomposition must be BIT-identical to the monomial one, since the only
    # term left is the leading one and degree 1 is raw on both.
    exponents = uniform_exponents(2, 2)
    coords = torch.stack([QUERY_X, torch.tensor([400.0, 700.0, 1000.0])], dim=1)

    monomial = build_seeded(exponents)
    legendre = build_with_bases(exponents, legendre_on_the_second_axis())
    for poly in (monomial, legendre):
        with torch.no_grad():
            poly.coefficients[0].zero_()

    assert torch.equal(monomial.evaluate(coords), legendre.evaluate(coords))


def test_the_gauge_exponent_is_zero_only_for_a_normalised_high_degree():
    # What renormalise_mode must undo per factor. Degree 1 scales like s on every
    # basis; a normalised factor above it does not move at all.
    assert MonomialBasis().gauge_exponent(3) == 3
    assert LegendreBasis().gauge_exponent(1) == 1
    assert LegendreBasis().gauge_exponent(2) == 0
