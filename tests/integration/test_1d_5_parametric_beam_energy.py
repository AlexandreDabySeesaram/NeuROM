"""Correctness tests for the 5-parametric (x, E1, E2, alpha, n) beam example.

The example script lives under docs/examples/ and its filename starts with a digit,
so it cannot be imported by module name; it is loaded from its path instead.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import torch

from neurom.differential import jacobian_field

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPO_ROOT
    / "docs"
    / "examples"
    / "1d_5-parametric_beam_PGD"
    / "1d_5-parametric_beam_deflection_PGD.py"
)


def load_module(path=EXAMPLE_PATH):
    """Import the example script from its path under an arbitrary module name."""
    spec = importlib.util.spec_from_file_location("beam5p", path)
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
    assert problem.pgd.n_modes_max == 10
    assert problem.pgd.n_modes_truncated == 1


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


def _flat(result_attr):
    return result_attr.reshape(-1)


def brute_force_energy(layout, decomposition, load_name="load", include_tanh=True):
    """Reference energy by direct 5-D tensor-product quadrature.

    ``include_tanh=False`` drops the tanh term from the modulus, leaving the
    constant modulus (E1 + E2) / 2 -- used to check the coupled block cancels
    when it must.
    """
    directory = decomposition.directory()
    n_modes = len(directory["space"])

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
    for i in range(n_modes):
        grad_u = grad_u + torch.einsum(
            "v,w,x,y,z->vwxyz", gX[i], lam[i], mu[i], A[i], N[i]
        )
        u_full = u_full + torch.einsum(
            "v,w,x,y,z->vwxyz", Xf[i], lam[i], mu[i], A[i], N[i]
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


@pytest.mark.parametrize("n_modes_ini", [1, 2])
def test_energy_matches_brute_force_5d_quadrature(beam5p, float64, n_modes_ini):
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_modes_ini=n_modes_ini,
        n_nodes=TINY,
    )
    _randomise_monoms(problem.pgd)

    layout = problem.model()
    separated = beam5p.energy(layout, problem.pgd)
    reference = brute_force_energy(layout, problem.pgd)

    assert separated.dim() == 0
    assert torch.isfinite(separated)
    assert separated.item() == pytest.approx(reference.item(), rel=1e-9)


def test_energy_is_differentiable_wrt_the_monoms(beam5p):
    problem = beam5p.build_problem(
        lambda layout, decomposition: beam5p.energy(layout, decomposition),
        n_modes_max=3,
        n_nodes=TINY,
    )
    layout = problem.model()
    loss = problem.model.loss(layout)
    loss.backward(retain_graph=True)

    for field in problem.pgd.monoms[0]:
        assert field.values_reduced.grad is not None
        assert torch.isfinite(field.values_reduced.grad).all()
        assert field.values_reduced.grad.abs().max() > 0.0


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
    problem = beam5p.main(verbose=True)

    assert problem.pgd.n_modes_truncated == 1
    assert problem.pgd.n_modes_max == 10

    layout = problem.model()
    value = problem.model.loss(layout)
    assert value.dim() == 0
    assert torch.isfinite(value)

    printed = capsys.readouterr().out
    match = re.search(r"energy\s*:\s*(\S+)", printed)
    assert match is not None, f"no 'energy : <value>' line found in output:\n{printed}"
    printed_energy = float(match.group(1))

    assert printed_energy == pytest.approx(value.item(), rel=1e-6)
