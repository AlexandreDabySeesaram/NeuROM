import pytest
import torch

from neurom.constraints import NoConstraint
from neurom.decompositions import CPPGD, MonomSpec
from neurom.field_layout import FieldLayout
from neurom.interpolation import IntegrationDomain
from neurom.math import integrate
from neurom.neurom_model import NeuROMModel
from neurom.shape_functions import LinearBar
from neurom.training import (
    FixedIterations,
    GreedyTrainer,
    MaxStages,
    Phase,
    RelativeChange,
)

torch.set_default_dtype(torch.float32)


# --- setup --------------------------------------------------------------
# L2 projection of the rank-2 target f(x, y) = x + 2y on [0, 1]^2:
#   J(u) = 1/2 int u^2 - int f u,   minimum -1/2 int f^2 = -4/3 at u = f.
# Not x + y: with identical seeds on both axes, a symmetric target keeps
# every mode symmetric (a_m = b_m), which cannot capture the residual.
# Separated, with u = sum_m a_m(x) b_m(y):
#   J = 1/2 sum_mn (int a_m a_n)(int b_m b_n)
#       - sum_m [(int x a_m)(int b_m) + 2 (int a_m)(int y b_m)].


@pytest.fixture
def make_model(make_factor_space):
    def _make_model():
        specs = [
            MonomSpec(
                space=make_factor_space(name=name, n=6, lo=0.0, hi=1.0),
                sf=LinearBar(),
                constraint=NoConstraint(),
                init_values=0.5 * torch.ones(6, 1),
            )
            for name in ("x", "y")
        ]
        cppgd = CPPGD(monom_specs=specs, n_modes_ini=1)

        def energy(layout):
            directory = cppgd.directory()
            a = [layout[name] for name in directory["x"]]
            b = [layout[name] for name in directory["y"]]

            def inner(f, g, h=None):
                w = f.u.values * g.u.values if g is not None else f.u.values
                if h is not None:
                    w = w * h
                return integrate(w * f.measure.values)

            n = len(a)
            quadratic = sum(
                inner(a[m], a[n_]) * inner(b[m], b[n_])
                for m in range(n)
                for n_ in range(n)
            )
            linear = sum(
                inner(a[m], None, a[m].x.values) * inner(b[m], None)
                + 2 * inner(a[m], None) * inner(b[m], None, b[m].x.values)
                for m in range(n)
            )
            return 0.5 * quadratic - linear

        return NeuROMModel(FieldLayout(), cppgd, IntegrationDomain([]), energy)

    return _make_model


def _adam(criterion, lr=0.1):
    return Phase(lambda p: torch.optim.Adam(p, lr=lr), criterion, "adam")


def _lbfgs(criterion):
    return Phase(
        lambda p: torch.optim.LBFGS(p, line_search_fn="strong_wolfe"),
        criterion,
        "lbfgs",
    )


def _values(cppgd, m):
    return [f.values_reduced.detach().clone() for f in cppgd.monoms[m]]


# --- greedy enrichment --------------------------------------------------


def test_one_mode_per_stage(make_model):
    model = make_model()
    trainer = GreedyTrainer(
        model,
        phases=[_adam(FixedIterations(5))],
        enrichment_criterion=MaxStages(3),
    )
    history = trainer.enrich()
    assert model.decomposition.n_modes == 3
    assert [s.stage for s in history.stages] == [0, 1, 2]
    assert [s.n_iter for s in history.stages] == [5, 5, 5]
    assert history.stop_reason == "n_stages"
    assert len(history.losses) == 15


def test_earlier_modes_do_not_move(make_model):
    model = make_model()
    trainer = GreedyTrainer(
        model,
        phases=[_adam(FixedIterations(20))],
        enrichment_criterion=MaxStages(1),
    )
    trainer.enrich()
    mode0 = _values(model.decomposition, 0)

    trainer.enrichment_criterion = MaxStages(2)
    trainer.enrich()  # resumes with stage 1
    assert len(trainer.history.stages) == 2
    for before, after in zip(mode0, _values(model.decomposition, 0)):
        assert torch.equal(before, after)


def test_energy_decreases_and_reaches_rank_two_minimum(make_model):
    model = make_model()
    trainer = GreedyTrainer(
        model,
        phases=[_lbfgs(RelativeChange(tol=1e-7, max_iter=50))],
        enrichment_criterion=MaxStages(2),
    )
    history = trainer.enrich()
    e0, e1 = (s.energy for s in history.stages)
    assert e1 < e0
    # Discrete L2 projection of a P1-representable target: exact up to round-off.
    assert e1 == pytest.approx(-4.0 / 3.0, rel=1e-5)


def test_matches_hand_written_greedy_loop(make_model):
    """The trainer reproduces the loop of examples/1d_bar_deflection_pgd."""
    n_iter, lr = 10, 0.1

    # Hand-written loop
    model = make_model()
    cppgd = model.decomposition
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def closure():
        optimizer.zero_grad()
        loss = model.loss(model())
        loss.backward()
        return loss

    expected = []
    for m in range(3):
        if m > 0:
            cppgd.freeze_mode(m - 1)
            cppgd.add_mode()
            cppgd.register_into(model.field_layout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_iter):
            expected.append(optimizer.step(closure).item())

    # Trainer
    trainer = GreedyTrainer(
        make_model(),
        phases=[_adam(FixedIterations(n_iter), lr=lr)],
        enrichment_criterion=MaxStages(3),
    )
    assert trainer.enrich().losses == expected


def test_diverged_stage_stops_the_run(make_model):
    model = make_model()
    trainer = GreedyTrainer(
        model,
        phases=[Phase(lambda p: torch.optim.SGD(p, lr=1e30), FixedIterations(50))],
        enrichment_criterion=MaxStages(3),
    )
    history = trainer.enrich()
    assert history.stop_reason == "diverged"
    assert history.stages[-1].stop_reason == "diverged"
    assert len(history.stages) == 1


def test_nothing_trainable_raises(make_model):
    model = make_model()
    trainer = GreedyTrainer(model)
    model.decomposition.freeze_all()
    with pytest.raises(RuntimeError):
        trainer.make_optimizer(lambda p: torch.optim.Adam(p))


# --- phases -------------------------------------------------------------


def test_phases_run_in_order_with_their_own_criterion(make_model):
    trainer = GreedyTrainer(
        make_model(),
        phases=[_adam(FixedIterations(7)), _lbfgs(FixedIterations(3))],
        enrichment_criterion=MaxStages(2),
    )
    history = trainer.enrich()
    for record in history.stages:
        assert record.phases == ["adam"] * 7 + ["lbfgs"] * 3
        assert record.n_iter == 10
        assert record.stop_reason == "n_iter"


def test_phase_already_met_is_skipped(make_model):
    trainer = GreedyTrainer(
        make_model(),
        phases=[_adam(FixedIterations(0)), _lbfgs(FixedIterations(2))],
    )
    assert trainer.enrich().stages[0].phases == ["lbfgs", "lbfgs"]


def test_lbfgs_after_adam_lowers_the_energy(make_model):
    adam_only = GreedyTrainer(make_model(), phases=[_adam(FixedIterations(20))])
    adam_then_lbfgs = GreedyTrainer(
        make_model(),
        phases=[_adam(FixedIterations(20)), _lbfgs(FixedIterations(5))],
    )
    e_adam = adam_only.enrich().stages[0].energy
    record = adam_then_lbfgs.enrich().stages[0]
    # Same Adam phase on the same seed: identical losses up to the switch.
    assert record.losses[:20] == adam_only.history.stages[0].losses
    assert record.energy < e_adam
