import pytest
import torch

from neurom.decompositions import CPPGD
from neurom.decompositions import TensorDecomposition
from neurom.field_layout import FieldLayout

torch.set_default_dtype(torch.float32)


# --- CPPGD construction -------------------------------------------------


def test_cppgd_builds_exactly_the_modes_asked_for(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)

    assert isinstance(model, TensorDecomposition)

    # Nothing is pre-allocated: 2 modes asked, 2 modes built, each with one
    # monom per factor.
    assert model.n_modes == 2
    assert len(model.monoms) == 2
    assert all(len(mode) == 2 for mode in model.monoms)
    assert len(model.assemblies()) == 2 * 2


def test_cppgd_monoms_are_born_trainable(two_specs):
    """The container applies no freezing policy of its own (that is the trainer's)."""
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)

    assert all(f.values_reduced.requires_grad for mode in model.monoms for f in mode)
    assert len([p for p in model.parameters() if p.requires_grad]) == 4


def test_two_vector_factors_raises(make_spec):
    a1 = make_spec(name="a", n=5, dim=2)
    a2 = make_spec(name="b", n=4, dim=3)
    with pytest.raises(ValueError):
        CPPGD(monom_specs=[a1, a2], n_modes_ini=1)


def test_cppgd_has_name_and_monom_naming(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2, name="beam")
    assert model.name == "beam"
    assert model.monoms[0][0].name == "beam_dimspace_mode0"
    assert model.monoms[1][1].name == "beam_dimE_mode1"


# --- Greedy mode lifecycle ----------------------------------------------


def test_add_mode_appends_a_row_to_the_grid(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    n_factors = len(two_specs)

    new = model.add_mode()

    assert new == 1
    assert model.n_modes == 2
    assert len(model.monoms) == 2
    assert len(model.assemblies()) == 2 * n_factors
    # New mode is trainable, previous one untouched.
    assert all(f.values_reduced.requires_grad for f in model.monoms[1])
    assert all(f.values_reduced.requires_grad for f in model.monoms[0])
    # New mode keeps its MonomSpec seed (no zero-out): an all-zero mode is a
    # stationary point of the energy and never takes off under a gradient
    # optimizer, so the seed is what lets the enrichment start moving.
    for f, spec in zip(model.monoms[1], two_specs):
        assert torch.equal(f.values_reduced, spec.init_values)


def test_add_mode_never_recomputes_geometry(two_specs):
    """A mode added later binds to its factor's existing QuadratureContext.

    This is what the FactorSpace / MonomSpec split buys: enrichment reads the
    blueprint again, it does not build a new discretisation.
    """
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    model.add_mode()

    n_factors = len(two_specs)
    flat = model.assemblies()
    for k, spec in enumerate(two_specs):
        assert flat[k].context is spec.space.context
        assert flat[n_factors + k].context is spec.space.context


def test_add_mode_is_unbounded(two_specs):
    """No capacity to hit: how far to enrich is not the container's business."""
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    for expected in range(1, 6):
        assert model.n_modes == expected
        model.add_mode()
    assert model.n_modes == 6


# --- Freezing: row, column and cell utilities ---------------------------


def _grad_grid(model):
    return [[f.values_reduced.requires_grad for f in mode] for mode in model.monoms]


def test_freeze_mode_freezes_a_row(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=3)
    model.freeze_mode(1)
    assert _grad_grid(model) == [[True, True], [False, False], [True, True]]
    model.unfreeze_mode(1)
    assert _grad_grid(model) == [[True, True]] * 3


def test_freeze_factor_freezes_a_column(two_specs):
    """What alternating-direction minimisation cycles over."""
    model = CPPGD(monom_specs=two_specs, n_modes_ini=3)
    model.freeze_factor(0)
    assert _grad_grid(model) == [[False, True]] * 3
    model.unfreeze_factor(0)
    model.freeze_factor(1)
    assert _grad_grid(model) == [[True, False]] * 3


def test_freeze_monom_freezes_a_single_cell(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)
    model.freeze_monom(1, 0)
    assert _grad_grid(model) == [[True, True], [False, True]]
    model.unfreeze_monom(1, 0)
    assert _grad_grid(model) == [[True, True], [True, True]]


def test_freeze_all_and_unfreeze_all(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)
    model.freeze_all()
    assert _grad_grid(model) == [[False, False]] * 2
    model.unfreeze_all()
    assert _grad_grid(model) == [[True, True]] * 2


def test_a_frozen_monom_is_still_interpolated(two_specs):
    """requires_grad says "not optimised", never "absent from u".

    Every monom that exists contributes to the energy, whatever its freeze
    state -- which is precisely why the old `active` flag was a different thing
    and could not serve this purpose.
    """
    from neurom.interpolation import IntegrationDomain

    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)
    model.freeze_mode(0)
    layout = FieldLayout()
    model.register_into(layout)

    IntegrationDomain([]).interpolate_all(layout, model.assemblies())

    for mode in model.monoms:
        for f in mode:
            assert layout[f.name].u.values is not None


def test_mode_parameters_returns_one_param_per_factor(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    model.add_mode()
    params = model.mode_parameters()  # defaults to the last mode added
    # one monom parameter per factor, and they are mode 1's tensors
    assert params == [
        model.monoms[1][0].values_reduced,
        model.monoms[1][1].values_reduced,
    ]


def test_mode_parameters_explicit_and_negative_index(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    model.add_mode()
    model.add_mode()

    assert model.mode_parameters(m=1) == [
        model.monoms[1][0].values_reduced,
        model.monoms[1][1].values_reduced,
    ]
    # Negative index resolves against the existing modes.
    assert model.mode_parameters(m=-1) == model.mode_parameters(m=2)


def test_mode_parameters_out_of_range_raises(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    # Only mode 0 exists.
    with pytest.raises(IndexError):
        model.mode_parameters(m=1)
    with pytest.raises(IndexError):
        model.mode_parameters(m=-2)


# --- Wiring into a FieldLayout / IntegrationDomain ----------------------


def test_register_into_populates_layout_with_all_monoms(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)
    layout = FieldLayout()
    model.register_into(layout)
    assert len(layout._fields) == 2 * 2
    for mode in model.monoms:
        for f in mode:
            assert f.name in layout._fields


def test_register_into_is_replayable_after_add_mode(two_specs):
    """How a new mode reaches the layout without the decomposition holding it."""
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    layout = FieldLayout()
    model.register_into(layout)

    model.add_mode()
    model.register_into(layout)  # idempotent on the modes already there

    assert len(layout._fields) == 2 * 2
    for f in model.monoms[1]:
        assert f.name in layout._fields


def test_directory_factor_major_names(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2, name="beam")
    d = model.directory()
    assert set(d.keys()) == {"space", "E"}
    assert d["space"] == ["beam_dimspace_mode0", "beam_dimspace_mode1"]
    assert d["E"] == ["beam_dimE_mode0", "beam_dimE_mode1"]
    model.add_mode()
    assert model.directory()["space"] == [
        "beam_dimspace_mode0",
        "beam_dimspace_mode1",
        "beam_dimspace_mode2",
    ]


def test_assemblies_accessor_is_flat_and_shares_factor_contexts(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=3)
    flat = model.assemblies()
    assert len(flat) == 3 * 2  # n_modes * n_factors
    # mode-major, factor order: block m, factor k -> flat[m * n_factors + k]
    assert flat[0].context is two_specs[0].space.context
    assert flat[1].context is two_specs[1].space.context
    assert flat[2].context is two_specs[0].space.context  # mode 1, factor 0


# --- Sampling the trained field: evaluate (diagonal) vs assemble (grid) --


def test_assemble_matches_manual_outer_product(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)

    # NoConstraint on both factors -> nodal values are the full field. Set them.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)  # S at 5 space nodes
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)  # g at 4 E nodes
        )

    # Space nodes are [0, 2.5, 5, 7.5, 10], E nodes are [100, 400, 700, 1000].
    # 2.5 and 400 sit ON a node; 6.25 and 550 sit at the MIDPOINT of an element,
    # so the linear interpolation is exercised, not just a nodal read-back.
    x = torch.tensor([2.5, 6.25])
    E = torch.tensor([400.0, 550.0])
    u = model.assemble([x, E])

    assert u.shape == (2, 2)

    # Hand-computed oracle, independent of the interpolator:
    #   S = [0, 1, 2, 3, 4]    -> S(2.5) = 1,   S(6.25) = (2 + 3) / 2 = 2.5
    #   g = [2, 3, 4, 5]       -> g(400) = 3,   g(550)  = (3 + 4) / 2 = 3.5
    #   u = S (x) g            -> [[3, 3.5], [7.5, 8.75]]
    expected = torch.tensor([[3.0, 3.5], [7.5, 8.75]])
    assert u.detach().numpy() == pytest.approx(expected.numpy(), rel=1e-5)


def test_assemble_sums_two_modes_matching_manual_outer_products(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=2)

    # NoConstraint on both factors -> nodal values are the full field. Set them
    # to distinct known vectors for each of the two modes.
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)  # S0 at 5 space nodes
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)  # g0 at 4 E nodes
        )
        model.monoms[1][0].values_reduced.copy_(
            torch.tensor([1.0, -1.0, 2.0, 0.5, -0.5]).unsqueeze(-1)  # S1
        )
        model.monoms[1][1].values_reduced.copy_(
            torch.tensor([-2.0, 1.0, 0.0, 3.0]).unsqueeze(-1)  # g1
        )

    # On a node (2.5, 400) and at an element midpoint (6.25, 550), see above.
    x = torch.tensor([2.5, 6.25])
    E = torch.tensor([400.0, 550.0])
    u = model.assemble([x, E])

    assert u.shape == (2, 2)

    # Hand-computed oracle, independent of the interpolator:
    #   S0 = [0, 1, 2, 3, 4]         -> S0(2.5) = 1,  S0(6.25) = 2.5
    #   g0 = [2, 3, 4, 5]            -> g0(400) = 3,  g0(550)  = 3.5
    #   S1 = [1, -1, 2, 0.5, -0.5]   -> S1(2.5) = -1, S1(6.25) = (2 + 0.5) / 2
    #                                                          = 1.25
    #   g1 = [-2, 1, 0, 3]           -> g1(400) = 1,  g1(550)  = (1 + 0) / 2
    #                                                          = 0.5
    #   S0 (x) g0 = [[3, 3.5], [7.5, 8.75]]
    #   S1 (x) g1 = [[-1, -0.5], [1.25, 0.625]]
    expected = torch.tensor([[2.0, 3.0], [8.75, 9.375]])
    assert u.detach().numpy() == pytest.approx(expected.numpy(), rel=1e-5)


def test_evaluate_matched_pointwise_matches_assemble_diagonal(two_specs):
    model = CPPGD(monom_specs=two_specs, n_modes_ini=1)
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.linspace(0.0, 4.0, 5).unsqueeze(-1)
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
        )
    # Same query points as the assemble tests above: one on a node, one at an
    # element midpoint.
    x = torch.tensor([2.5, 6.25])
    E = torch.tensor([400.0, 550.0])
    u = model.evaluate(torch.stack([x, E], dim=1))  # matched, (2, 1)
    grid = model.assemble([x, E])  # (2, 2)
    assert u.shape == (2, 1)
    assert u.reshape(-1).detach().numpy() == pytest.approx(
        torch.diagonal(grid).detach().numpy(), rel=1e-5
    )


def test_evaluate_and_assemble_vector_factor(make_spec):
    space = make_spec(name="space", n=5, dim=2)  # 2-D displacement factor
    para = make_spec(name="E", n=4, lo=100.0, hi=1000.0)  # scalar weight
    model = CPPGD(monom_specs=[space, para], n_modes_ini=1)
    with torch.no_grad():
        model.monoms[0][0].values_reduced.copy_(
            torch.arange(10, dtype=torch.float32).reshape(5, 2)
        )
        model.monoms[0][1].values_reduced.copy_(
            torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(-1)
        )
    # On a node (2.5, 400) and at an element midpoint (6.25, 550).
    x = torch.tensor([2.5, 6.25])
    E = torch.tensor([400.0, 550.0])

    u = model.evaluate(torch.stack([x, E], dim=1))
    assert u.shape == (2, 2)  # (P, d)
    grid = model.assemble([x, E])
    assert grid.shape == (2, 2, 2)  # (N_x, N_E, d)

    # Hand-computed oracle, independent of the interpolator. The vector factor
    # holds [0, 1], [2, 3], [4, 5], [6, 7], [8, 9] at the five space nodes:
    #   S(2.5)  = [2, 3]                  g(400) = 3
    #   S(6.25) = ([4, 5] + [6, 7]) / 2   g(550) = 3.5
    #           = [5, 6]
    # evaluate() pairs them row-wise: [2, 3] * 3 and [5, 6] * 3.5.
    expected_diag = torch.tensor([[6.0, 9.0], [17.5, 21.0]])
    assert u.detach().numpy() == pytest.approx(expected_diag.numpy(), rel=1e-5)

    # assemble() crosses every (x, E) pair; its diagonal is what evaluate()
    # returns, and the off-diagonal terms mix the two.
    expected_grid = torch.tensor(
        [
            [[6.0, 9.0], [7.0, 10.5]],  # S(2.5)  * g(400), S(2.5)  * g(550)
            [[15.0, 18.0], [17.5, 21.0]],  # S(6.25) * g(400), S(6.25) * g(550)
        ]
    )
    assert grid.detach().numpy() == pytest.approx(expected_grid.numpy(), rel=1e-5)
