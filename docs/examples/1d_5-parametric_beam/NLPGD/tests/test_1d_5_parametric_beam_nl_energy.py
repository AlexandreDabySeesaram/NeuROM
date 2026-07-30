"""Correctness tests for the polynomial NL-PGD 5-parametric beam example.

The example script lives under docs/examples/ and its filename starts with a digit,
so it cannot be imported by module name; it is loaded from its path instead.

The load-bearing test here is
:func:`test_energy_matches_brute_force_5d_quadrature`, parametrised over
**zero and non-zero coefficients**. At ``C = 0`` it re-checks the CP energy the
sibling example already pins; at ``C != 0`` it is the only thing standing between
the separated polynomial energy and a wrong answer that still trains to a
plausible-looking figure.
"""

import importlib.util
import math
import re
from pathlib import Path

import pytest
import torch

from neurom.differential import jacobian_field
from neurom.quadratures import MidPoint1D, TwoPoints1D

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = EXAMPLE_DIR / "1d_5-parametric_beam_deflection_NLPGD.py"


def load_module(path=EXAMPLE_PATH):
    """Import the example script from its path under an arbitrary module name."""
    spec = importlib.util.spec_from_file_location("beam5nl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def beam5p():
    return load_module()


def test_axes_are_the_five_expected_coordinates(beam5p):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    assert [axis.name for axis in problem.pgd.axes] == [
        "space",
        "E1",
        "E2",
        "alpha",
        "n",
    ]
    assert list(problem.pgd.directory()) == ["space", "E1", "E2", "alpha", "n"]
    assert problem.pgd.n_modes_max == 5
    assert problem.pgd.n_modes_truncated == 1


def test_the_default_decomposition_is_polynomial_but_starts_frozen_at_zero(beam5p):
    """The default exponent set, and the invariant every baseline run relies on.

    A freshly built problem must be numerically indistinguishable from the CP
    one: every coefficient row zero AND frozen. If either half regressed, the
    ``greedy`` / ``simultaneous`` strategies would stop being a CP baseline and
    the comparison this whole directory exists for would be meaningless.
    """
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    assert problem.pgd.exponents.tolist() == [[2] * 5, [3] * 5]
    assert problem.pgd.n_terms == 2
    for row in problem.pgd.coefficients:
        assert not row.requires_grad
        assert torch.equal(row, torch.zeros_like(row))


def test_polynomial_directory_enumerates_leading_then_correction_terms(beam5p):
    problem = beam5p.build_problem(
        lambda layout, decomposition: torch.zeros(()), n_modes_max=3, n_modes_ini=2
    )

    terms = problem.pgd.polynomial_directory()

    assert [(m, lam) for m, lam, _ in terms] == [
        (0, (1, 1, 1, 1, 1)), (0, (2,) * 5), (0, (3,) * 5),
        (1, (1, 1, 1, 1, 1)), (1, (2,) * 5), (1, (3,) * 5),
    ]
    # The leading term's weight is a fixed 1, not a parameter.
    assert [coefficient is None for _, _, coefficient in terms][::3] == [True, True]


@pytest.mark.parametrize(
    "exponent_set, max_power, expected",
    [
        ("uniform", 2, [[2] * 5]),
        ("uniform", 4, [[2] * 5, [3] * 5, [4] * 5]),
        # All exponents >= 1 summing to exactly 6: the five permutations of
        # (2,1,1,1,1). (1,1,1,1,1) is the leading term and excluded.
        ("total_degree", 6, [
            [1, 1, 1, 1, 2], [1, 1, 1, 2, 1], [1, 1, 2, 1, 1],
            [1, 2, 1, 1, 1], [2, 1, 1, 1, 1],
        ]),
    ],
)
def test_build_exponents_maps_the_config_knob(beam5p, exponent_set, max_power, expected):
    cfg = beam5p.RunConfig(name="x", exponent_set=exponent_set, max_power=max_power)

    assert sorted(beam5p.build_exponents(cfg).tolist()) == sorted(expected)


def test_build_exponents_rejects_an_unknown_set(beam5p):
    cfg = beam5p.RunConfig(name="x", exponent_set="nonsense")

    with pytest.raises(ValueError, match="unknown exponent_set"):
        beam5p.build_exponents(cfg)


@pytest.mark.parametrize(
    "name, lo, hi, n_nodes",
    [
        ("space", 0.0, 10.0, 30),
        ("E1", 10.0, 100.0, 20),
        ("E2", 10.0, 100.0, 20),
        ("alpha", 2.0, 8.0, 15),
        ("n", 0.5, 5.0, 15),
    ],
)
def test_axis_intervals_and_resolutions(beam5p, name, lo, hi, n_nodes):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    positions = problem.axes[name].nodes_positions.values
    assert positions.shape == (n_nodes, 1)
    assert positions.min().item() == pytest.approx(lo)
    assert positions.max().item() == pytest.approx(hi)


def test_forward_interpolates_every_monom_and_the_load(beam5p):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    layout = problem.model()

    # MidPoint1D: one quadrature point per element, n_nodes - 1 elements.
    expected = {"space": 29, "E1": 19, "E2": 19, "alpha": 14, "n": 14}
    directory = problem.pgd.directory()
    for axis_name, n_elements in expected.items():
        result = layout[directory[axis_name][0]]
        assert result.u.shape == (n_elements, 1, 1)
        assert result.x.shape == (n_elements, 1, 1)
        assert result.measure.shape == (n_elements, 1, 1)

    # The load must be sampled on the *same* quadrature as the space monom,
    # otherwise inner(load, X) silently broadcasts wrong.
    assert layout["load"].u.shape == layout[directory["space"][0]].u.shape
    assert torch.allclose(layout["load"].u, torch.full_like(layout["load"].u, 1000.0))


def test_n_nodes_override_builds_a_smaller_problem(beam5p):
    problem = beam5p.build_problem(
        lambda layout, decomposition: torch.zeros(()),
        n_nodes={"space": 5, "E1": 4, "E2": 4, "alpha": 4, "n": 4},
    )

    layout = problem.model()
    directory = problem.pgd.directory()
    assert layout[directory["space"][0]].u.shape == (4, 1, 1)
    assert layout[directory["n"][0]].u.shape == (3, 1, 1)


def test_loss_is_the_injected_callable(beam5p):
    seen = {}

    def spy(layout, decomposition):
        seen["layout"] = layout
        seen["decomposition"] = decomposition
        return torch.tensor(42.0)

    problem = beam5p.build_problem(spy)
    out = problem.model()

    assert problem.model.loss(out).item() == 42.0
    assert seen["layout"] is problem.field_layout
    assert seen["decomposition"] is problem.pgd


# --- brute-force reference ------------------------------------------------
#
# Independent check of the separated energy: assemble u and grad_x u as FULL
# 5-D tensors over the tensor product of every axis's quadrature points, form
# E(x, E1, E2, alpha, n) pointwise there, and sum. This exploits no separability
# whatsoever, so it cannot share a bug with the implementation under test. Only
# tractable because the test uses a tiny mesh (4 x 3 x 3 x 3 x 3 = 324 points).
#
# Adapted from the CP version by summing over TERMS rather than modes, and by
# applying the chain rule to each term's space factor. The two differences are
# exactly the two places the polynomial energy could go wrong, so the reference
# deliberately spells them out in the dumbest possible way -- one Python loop
# over `polynomial_directory()`, no factorisation, no hoisting.


def _flat(result_attr):
    return result_attr.reshape(-1)


def brute_force_energy(layout, decomposition, load_name="load", include_tanh=True):
    """Reference energy by direct 5-D tensor-product quadrature.

    ``include_tanh=False`` drops the tanh term from the modulus, leaving the
    constant modulus (E1 + E2) / 2 -- used to check the coupled block cancels
    when it must.
    """
    directory = decomposition.directory()

    spc = [layout[name] for name in directory["space"]]
    e1 = [layout[name] for name in directory["E1"]]
    e2 = [layout[name] for name in directory["E2"]]
    alp = [layout[name] for name in directory["alpha"]]
    slp = [layout[name] for name in directory["n"]]

    gX = [_flat(jacobian_field(x=r.x, u=r.u)) for r in spc]
    Xf = [_flat(r.u) for r in spc]
    lam = [_flat(r.u) for r in e1]
    mu = [_flat(r.u) for r in e2]
    A = [_flat(r.u) for r in alp]
    N = [_flat(r.u) for r in slp]

    xq, aq, nq = _flat(spc[0].x), _flat(alp[0].x), _flat(slp[0].x)
    e1q, e2q = _flat(e1[0].x), _flat(e2[0].x)

    shape = (xq.numel(), e1q.numel(), e2q.numel(), aq.numel(), nq.numel())
    grad_u = torch.zeros(shape, dtype=xq.dtype)
    u_full = torch.zeros(shape, dtype=xq.dtype)
    for i, lamb, coefficient in decomposition.polynomial_directory():
        weight = 1.0 if coefficient is None else coefficient
        p = lamb[0]
        # d/dx [ X^p ] = p X^(p-1) X' -- the chain-rule factor the CP version
        # has no need of (p == 1 there, so the factor is 1).
        grad_u = grad_u + weight * p * torch.einsum(
            "v,w,x,y,z->vwxyz",
            Xf[i] ** (p - 1) * gX[i],
            lam[i] ** lamb[1],
            mu[i] ** lamb[2],
            A[i] ** lamb[3],
            N[i] ** lamb[4],
        )
        u_full = u_full + weight * torch.einsum(
            "v,w,x,y,z->vwxyz",
            Xf[i] ** p,
            lam[i] ** lamb[1],
            mu[i] ** lamb[2],
            A[i] ** lamb[3],
            N[i] ** lamb[4],
        )

    xv = xq.view(-1, 1, 1, 1, 1)
    e1v = e1q.view(1, -1, 1, 1, 1)
    e2v = e2q.view(1, 1, -1, 1, 1)
    av = aq.view(1, 1, 1, -1, 1)
    nv = nq.view(1, 1, 1, 1, -1)

    E_grid = 0.5 * (e2v + e1v)
    if include_tanh:
        E_grid = E_grid + 0.5 * (e2v - e1v) * torch.tanh(nv * (xv - av))

    measure = (
        _flat(spc[0].measure).view(-1, 1, 1, 1, 1)
        * _flat(e1[0].measure).view(1, -1, 1, 1, 1)
        * _flat(e2[0].measure).view(1, 1, -1, 1, 1)
        * _flat(alp[0].measure).view(1, 1, 1, -1, 1)
        * _flat(slp[0].measure).view(1, 1, 1, 1, -1)
    )

    fq = _flat(layout[load_name].u).view(-1, 1, 1, 1, 1)

    elastic = 0.5 * torch.sum(E_grid * grad_u * grad_u * measure)
    load = torch.sum(fq * u_full * measure)
    return elastic + load


# Every axis has a distinct node count, so an axis mix-up (e.g. reading E2's
# values/measure where E1's were meant, or swapping alpha/n in an einsum)
# turns into a shape mismatch instead of a silently-passing wrong number.
TINY = {"space": 5, "E1": 4, "E2": 6, "alpha": 7, "n": 3}

# test_flat_modulus_limit_matches_the_separable_only_energy needs E1 and E2 to
# share a mesh (same node count, same interval) for its cancellation argument
# to hold, so it gets its own dict with E1 == E2.
TINY_FLAT = {"space": 5, "E1": 4, "E2": 4, "alpha": 7, "n": 3}


@pytest.fixture
def float64():
    """Run the energy comparison in double precision.

    The separated form and the brute-force form sum in very different orders and
    the elastic term involves a difference (M1 L0 - L1 M0); float32 leaves too
    little margin to distinguish a real bug from round-off.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _randomise_monoms(pgd, seed=0):
    """Give every active monom non-trivial nodal values, spanning both signs.

    Strictly positive values (e.g. ``rand + 0.5``) would never exercise
    sign-dependent cancellation paths, so this draws from ``[-1, 1)`` instead.
    """
    generator = torch.Generator().manual_seed(seed)
    for m in range(pgd.n_modes_truncated):
        for field in pgd.monoms[m]:
            with torch.no_grad():
                field.values_reduced.copy_(
                    2.0
                    * torch.rand(
                        field.values_reduced.shape,
                        generator=generator,
                        dtype=field.values_reduced.dtype,
                    )
                    - 1.0
                )


def _randomise_coefficients(pgd, seed=1, scale=0.3):
    """Give every active mode a non-trivial coefficient row, spanning both signs.

    ``scale`` is deliberately small. The correction terms carry fifth powers of
    the monoms, so an O(1) coefficient makes the energy a product of very
    different magnitudes and the float64 comparison starts measuring round-off
    rather than correctness.
    """
    generator = torch.Generator().manual_seed(seed)
    for m in range(pgd.n_modes_truncated):
        with torch.no_grad():
            pgd.coefficients[m].copy_(
                scale
                * (
                    2.0
                    * torch.rand(
                        pgd.coefficients[m].shape,
                        generator=generator,
                        dtype=pgd.coefficients[m].dtype,
                    )
                    - 1.0
                )
            )


@pytest.mark.parametrize("quad_cls", [MidPoint1D, TwoPoints1D])
@pytest.mark.parametrize("n_modes_ini", [1, 2])
@pytest.mark.parametrize("with_coefficients", [False, True])
def test_energy_matches_brute_force_5d_quadrature(
    beam5p, float64, n_modes_ini, quad_cls, with_coefficients
):
    # The central correctness test of this example. `with_coefficients=False` is
    # the CP case (C = 0, the polynomial terms must contribute nothing);
    # `True` is the case the separated polynomial energy exists for. Both are run
    # against the same brute-force 5-D quadrature.
    #
    # The quadrature rule is instantiated inside the test (not at
    # parametrize-decoration time) so its buffers pick up the float64 default the
    # `float64` fixture has already switched to; building it at collection time
    # would freeze it at whatever dtype was active at import, which mismatches
    # the rest of the (float64) problem and fails with a dtype error in einsum.
    quad = quad_cls()
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=n_modes_ini,
        n_nodes=TINY,
        quad=quad,
    )
    _randomise_monoms(problem.pgd)
    if with_coefficients:
        _randomise_coefficients(problem.pgd)

    layout = problem.model()
    separated = beam5p.energy(layout, problem.pgd)
    reference = brute_force_energy(layout, problem.pgd)

    assert separated.dim() == 0
    assert torch.isfinite(separated)
    assert separated.item() == pytest.approx(reference.item(), rel=1e-9)


def test_energy_after_add_mode_matches_brute_force(beam5p, float64):
    """Pin the greedy-enrichment seam: energy must stay correct after add_mode.

    ``add_mode`` activates a new mode's pre-allocated assemblies without
    touching the ``IntegrationDomain`` or rebuilding it, and ``energy`` must
    pick the new mode up through ``directory()`` alone. This is exactly the
    machinery the next piece of work (greedy training) turns on, and nothing
    else exercises it yet.
    """
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_nodes=TINY,
    )

    problem.pgd.add_mode()
    assert problem.pgd.n_modes_truncated == 2
    _randomise_monoms(problem.pgd)
    # The new mode's correction terms must be picked up too, not only its
    # leading term: polynomial_directory() reads n_modes_truncated the same way
    # directory() does, and this is what proves the two stayed in step.
    _randomise_coefficients(problem.pgd)

    layout = problem.model()
    separated = problem.model.loss(layout)
    reference = brute_force_energy(layout, problem.pgd)

    assert torch.isfinite(separated)
    assert separated.item() == pytest.approx(reference.item(), rel=1e-9)


def test_energy_is_differentiable_wrt_the_monoms(beam5p):
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=2,
        n_nodes=TINY,
    )
    layout = problem.model()
    loss = problem.model.loss(layout)
    loss.backward(retain_graph=True)

    for m in range(problem.pgd.n_modes_truncated):
        for field in problem.pgd.monoms[m]:
            assert field.values_reduced.grad is not None
            assert torch.isfinite(field.values_reduced.grad).all()
            assert field.values_reduced.grad.abs().max() > 0.0


def test_energy_is_differentiable_wrt_the_coefficients(beam5p):
    """A released coefficient row must receive a non-zero gradient.

    The reason this needs its own test: ``C`` never reaches a
    ``QuadratureAssembly`` and is not a ``Field``, so it rides into the graph
    only through the arithmetic in ``energy``. A version of ``energy`` that read
    the exponents but dropped the coefficients would still produce a finite
    number, still train the monoms, and differ from CP-PGD by nothing at all.
    """
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=2,
        n_nodes=TINY,
    )
    _randomise_monoms(problem.pgd)
    for m in range(problem.pgd.n_modes_truncated):
        problem.pgd.unfreeze_mode_coefficients(m)

    problem.model.loss(problem.model()).backward(retain_graph=True)

    for m in range(problem.pgd.n_modes_truncated):
        grad = problem.pgd.coefficients[m].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().max() > 0.0


def test_zero_coefficients_reproduce_the_cp_energy_exactly(beam5p, float64):
    """The invariant the CP baseline strategies rest on, on the real energy.

    With every ``C = 0`` the polynomial terms must contribute *nothing* -- not
    "a little", exactly nothing. Checked against the brute-force reference with
    the term list truncated to the leading terms, which is the CP field by
    definition. If this drifts, comparing a ``greedy`` row against a
    ``polynomial`` row in the ledger stops meaning anything.
    """
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=2,
        n_nodes=TINY,
    )
    _randomise_monoms(problem.pgd)
    layout = problem.model()

    with_polynomial = beam5p.energy(layout, problem.pgd)

    # A stand-in decomposition whose term list is the leading terms alone: the
    # CP decomposition of the very same monoms.
    class _LeadingTermsOnly:
        def __init__(self, pgd):
            self._pgd = pgd

        def directory(self):
            return self._pgd.directory()

        def polynomial_directory(self):
            return [
                term for term in self._pgd.polynomial_directory() if term[2] is None
            ]

    cp_only = beam5p.energy(layout, _LeadingTermsOnly(problem.pgd))

    assert with_polynomial.item() == cp_only.item()


def test_flat_modulus_limit_matches_the_separable_only_energy(beam5p, float64):
    """When the E1 and E2 factors coincide, the whole tanh block must drop out.

    The coupled term carries the prefactor (M1 L0 - L1 M0), the discrete image of
    (E2 - E1)/2. The two modulus axes share an interval and a mesh here, so giving
    them identical monom values makes M1 == L1 and L0 == M0, and that prefactor
    vanishes exactly. The energy must then equal the brute-force reference
    computed with the tanh term dropped entirely -- if it does not, the coupled
    and separable parts are wired together wrongly.

    n_modes_ini=2 keeps both modes active (not just mode 0), and the E1->E2
    copy runs for every active mode, so the cancellation is exercised across
    all mode pairs, including the (0, 1) / (1, 0) cross terms.
    """
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=2,
        n_modes_ini=2,
        n_nodes=TINY_FLAT,
    )
    _randomise_monoms(problem.pgd)
    # Identical E1 and E2 factors on every active mode: same mesh, same
    # interval, same nodal values.
    with torch.no_grad():
        for m in range(problem.pgd.n_modes_truncated):
            problem.pgd.monoms[m][2].values_reduced.copy_(
                problem.pgd.monoms[m][1].values_reduced
            )

    layout = problem.model()
    separated = beam5p.energy(layout, problem.pgd)
    flat_reference = brute_force_energy(layout, problem.pgd, include_tanh=False)

    assert separated.item() == pytest.approx(flat_reference.item(), rel=1e-9)


def test_main_builds_and_evaluates_a_finite_energy(beam5p, capsys):
    problem = beam5p.main(verbose=True, train=False)

    assert problem.pgd.n_modes_truncated == 1
    assert problem.pgd.n_modes_max == 5

    layout = problem.model()
    value = problem.model.loss(layout)
    assert value.dim() == 0
    assert torch.isfinite(value)

    printed = capsys.readouterr().out
    match = re.search(r"energy\s*:\s*(\S+)", printed)
    assert match is not None, f"no 'energy : <value>' line found in output:\n{printed}"
    printed_energy = float(match.group(1))

    assert printed_energy == pytest.approx(value.item(), rel=1e-6)


def test_main_trains_and_reports_a_decreasing_energy(beam5p, capsys):
    """``main``'s whole train-and-report path, on a deliberately tiny config.

    Two departures from the CP example's version of this test, both deliberate:

    * A **tiny config** rather than the shipped defaults. The polynomial energy's
      double loop is quadratic in ``n_modes * (1 + |I|)``, so a full-size run is
      minutes, not seconds. This test is about the wiring -- does the trainer
      drive ``main``, does the table print -- not about accuracy.
    * ``checkpoint=False``. Without it, ``main`` loads whatever happens to be in
      ``param_sweep/`` and the test silently stops training anything at all,
      asserting against a stale run from an unrelated configuration.
    """
    config = beam5p.RunConfig(
        name="test-tiny",
        strategy="joint",
        stage_tol=1e-3, window=5, max_iter=12, min_iter=6,
        enrichment_tol=1e-6, n_modes_max=3,
        n_nodes={"space": 6, "E1": 4, "E2": 4, "alpha": 4, "n": 4},
    )
    problem = beam5p.main(
        verbose=True, train=True, plot=False, config=config, checkpoint=False
    )

    assert problem.history is not None

    # Enrichment must actually get past mode 0 -- that is the point of wiring the
    # trainer in. The stage-to-mode ratio is the schedule's `stages_per_mode`,
    # which is 1 for "joint" and 2 for "staged"/"refine"; the CP example's
    # unconditional `n_modes == len(stages)` does not carry over.
    assert problem.pgd.n_modes_truncated > 1
    # The "joint" schedule is one stage per mode.
    assert len(problem.history.stages) == problem.pgd.n_modes_truncated
    kinds = [r.diagnostics["kind"] for r in problem.history.stages]
    assert kinds == ["joint"] * problem.pgd.n_modes_truncated

    # Each stage must pay for itself, not just the run as a whole.
    for record in problem.history.stages[1:]:
        assert record.energy <= problem.history.stages[0].energy

    losses = problem.history.losses
    assert losses[-1] < losses[0]
    assert all(math.isfinite(value) for value in losses)

    printed = capsys.readouterr().out
    assert "stage" in printed
    # "stage" alone only matches the table header -- a broken table body
    # (e.g. an exception mid-loop, or an empty stages list) would still pass.
    # Also require a printed row for the last stage, so the body is checked.
    last_stage = problem.history.stages[-1]
    assert f"{last_stage.stage:5d}" in printed
    # The coefficient table is the one piece of output no CP run produces.
    assert "polynomial coefficients C" in printed


# --- the leading coefficient and the pinned space exponent ------------------


@pytest.mark.parametrize("quad_cls", [MidPoint1D, TwoPoints1D])
@pytest.mark.parametrize("n_modes_ini", [1, 2])
def test_energy_matches_brute_force_with_a_non_unit_leading_coefficient(
    beam5p, float64, n_modes_ini, quad_cls
):
    """``c != 1`` is a path nothing else reaches.

    ``energy`` folds each term's weight into **two** hoisted quantities -- the
    chain-rule prefactor ``gpref`` and the value ``xval`` -- and a leading term
    has always carried ``weight = 1.0``, so a bug in either (a dropped factor, a
    factor applied twice) is invisible while the weight is 1. Against the same
    brute-force 5-D quadrature, which reads the weight generically from
    ``polynomial_directory()``.
    """
    quad = quad_cls()
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=n_modes_ini,
        n_nodes=TINY,
        quad=quad,
        leading_coefficients=True,
    )
    _randomise_monoms(problem.pgd)
    _randomise_coefficients(problem.pgd)
    with torch.no_grad():
        for m in range(problem.pgd.n_modes_truncated):
            problem.pgd.leading_coefficients[m].fill_(0.7 + 0.5 * m)

    layout = problem.model()
    separated = beam5p.energy(layout, problem.pgd)
    reference = brute_force_energy(layout, problem.pgd)

    assert torch.isfinite(separated)
    assert separated.item() == pytest.approx(reference.item(), rel=1e-9)


def test_a_unit_leading_coefficient_reproduces_the_fixed_weight_energy(
    beam5p, float64
):
    """c = 1 must be bitwise the fixed-weight energy, not merely close."""
    energies = []
    for leading in (False, True):
        problem = beam5p.build_problem(
            lambda layout, decomposition: beam5p.energy(layout, decomposition),
            n_modes_max=2,
            n_nodes=TINY,
            leading_coefficients=leading,
        )
        _randomise_monoms(problem.pgd)
        _randomise_coefficients(problem.pgd)
        energies.append(beam5p.energy(problem.model(), problem.pgd))

    assert energies[0].item() == energies[1].item()


@pytest.mark.parametrize("quad_cls", [MidPoint1D, TwoPoints1D])
def test_energy_matches_brute_force_with_the_space_exponent_pinned(
    beam5p, float64, quad_cls
):
    """A pinned space exponent drives ``p = 1``, so ``gpref`` becomes ``X ** 0``.

    Every other exponent set in this file has ``p >= 2`` on every correction
    term, so the ``p - 1 == 0`` branch of the chain-rule prefactor is otherwise
    never taken -- and it is the one where a wrong power silently evaluates to
    an array of ones instead of erroring.
    """
    cfg = beam5p.RunConfig(name="pinned", pin_space_exponent=True)
    exponents = beam5p.build_exponents(cfg)
    assert all(int(row[0]) == 1 for row in exponents)

    quad = quad_cls()
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=2,
        n_nodes=TINY,
        quad=quad,
        exponents=exponents,
    )
    _randomise_monoms(problem.pgd)
    _randomise_coefficients(problem.pgd)

    layout = problem.model()
    separated = beam5p.energy(layout, problem.pgd)
    reference = brute_force_energy(layout, problem.pgd)

    assert torch.isfinite(separated)
    assert separated.item() == pytest.approx(reference.item(), rel=1e-9)
