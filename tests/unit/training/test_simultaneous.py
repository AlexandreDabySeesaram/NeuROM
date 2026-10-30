import pytest
import torch

from neurom.training import (
    AnyOf,
    FixedIterations,
    GreedyTrainer,
    MaxStages,
    Phase,
    SimultaneousTrainer,
    Stagnation,
    Usefulness,
)

torch.set_default_dtype(torch.float32)

# The toy model `make_model` (rank-2 L2 projection, minimum -4/3) is in conftest.py.


def _adam(criterion):
    return Phase(lambda p: torch.optim.Adam(p, lr=0.1), criterion, "adam")


def _lbfgs(criterion):
    return Phase(lambda p: torch.optim.LBFGS(p, line_search_fn="strong_wolfe"), criterion, "lbfgs")


def _values(cppgd, m):
    return [f.values_reduced.detach().clone() for f in cppgd.monoms[m]]


def _two_stages(make_model, trainer_class):
    """Run one stage, then resume for a second; return the trainer and mode 0 before/after."""
    model = make_model()
    trainer = trainer_class(model, phases=[_adam(FixedIterations(20))], enrichment_criterion=MaxStages(1))
    trainer.enrich()
    before = _values(model.decomposition, 0)
    trainer.enrichment_criterion = MaxStages(2)
    trainer.enrich()  # resumes with stage 1
    return trainer, before, _values(model.decomposition, 0)


def test_one_mode_per_stage(make_model):
    model = make_model()
    history = SimultaneousTrainer(
        model, phases=[_adam(FixedIterations(5))], enrichment_criterion=MaxStages(3)
    ).enrich()
    assert model.decomposition.n_modes == 3
    assert [s.n_iter for s in history.stages] == [5, 5, 5]
    assert history.stop_reason == "n_stages"


def test_earlier_modes_move(make_model):
    _, before, after = _two_stages(make_model, SimultaneousTrainer)
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    # Same schedule, greedy: mode 0 is frozen.
    _, before, after = _two_stages(make_model, GreedyTrainer)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_the_optimizer_is_kept_across_stages(make_model):
    trainer, _, _ = _two_stages(make_model, SimultaneousTrainer)
    optimizer = trainer.optimizer
    assert len(optimizer.param_groups) == 2  # mode 1 added as a group
    mode0 = trainer.model.decomposition.mode_parameters(0)[0]
    assert int(optimizer.state[mode0]["step"]) == 40  # Adam's state ran on
    mode1 = trainer.model.decomposition.mode_parameters(1)[0]
    assert int(optimizer.state[mode1]["step"]) == 20
    # Greedy builds a new optimizer per stage: mode 0 is not in it.
    trainer, _, _ = _two_stages(make_model, GreedyTrainer)
    assert mode0 not in trainer.optimizer.state


def test_unfreezes_a_frozen_decomposition(make_model):
    model = make_model()
    model.decomposition.freeze_all()
    trainer = SimultaneousTrainer(model, phases=[_adam(FixedIterations(2))])
    trainer.enrich()  # would raise "No trainable parameters" if left frozen
    assert all(f.values_reduced.requires_grad for f in model.decomposition.monoms[0])


def test_takes_exactly_one_phase(make_model):
    with pytest.raises(ValueError, match="exactly one phase"):
        SimultaneousTrainer(
            make_model(), phases=[_adam(FixedIterations(2)), _lbfgs(FixedIterations(2))]
        )


def test_rejects_lbfgs_as_the_stage_optimizer(make_model):
    trainer = SimultaneousTrainer(make_model(), phases=[_lbfgs(FixedIterations(2))])
    with pytest.raises(ValueError, match="final_phases"):
        trainer.enrich()



# --- keep_optimizer=False -----------------------------------------------


def test_fresh_optimizer_trains_every_mode_with_a_new_one_per_stage(make_model):
    trainer = SimultaneousTrainer(
        make_model(),
        keep_optimizer=False,
        phases=[_adam(FixedIterations(20))],
        enrichment_criterion=MaxStages(2),
    )
    trainer.enrich()
    decomposition = trainer.model.decomposition
    mode0 = decomposition.mode_parameters(0)[0]
    assert int(trainer.optimizer.state[mode0]["step"]) == 20  # restarted at stage 1
    mode1 = decomposition.mode_parameters(1)[0]
    assert int(trainer.optimizer.state[mode1]["step"]) == 20


def test_fresh_optimizer_allows_an_lbfgs_per_stage(make_model):
    history = SimultaneousTrainer(
        make_model(),
        keep_optimizer=False,
        phases=[_lbfgs(Stagnation(tol=1e-7, patience=0, no_improvement=None, max_iter=50))],
        enrichment_criterion=MaxStages(2),
    ).enrich()
    assert history.stages[-1].energy == pytest.approx(-4.0 / 3.0, rel=1e-5)


def test_fresh_optimizer_allows_several_phases_per_stage(make_model):
    history = SimultaneousTrainer(
        make_model(),
        keep_optimizer=False,
        phases=[_adam(FixedIterations(5)), _lbfgs(FixedIterations(3))],
        enrichment_criterion=MaxStages(2),
    ).enrich()
    for record in history.stages:
        assert record.phases == ["adam"] * 5 + ["lbfgs"] * 3

def test_final_lbfgs_reaches_the_rank_two_minimum(make_model):
    model = make_model()
    history = SimultaneousTrainer(
        model,
        phases=[_adam(FixedIterations(50))],
        enrichment_criterion=MaxStages(2),
        final_phases=[_lbfgs(Stagnation(tol=1e-7, patience=0, no_improvement=None, max_iter=50))],
    ).enrich()
    assert model.decomposition.n_modes == 2
    assert history.final.energy == pytest.approx(-4.0 / 3.0, rel=1e-5)


def test_algorithm_1_stops_by_itself(make_model):
    history = SimultaneousTrainer(
        make_model(),
        phases=[_adam(Stagnation(tol=1e-4))],
        enrichment_criterion=AnyOf(Usefulness(tol=1e-4), MaxStages(8)),
    ).enrich()
    assert {s.stop_reason for s in history.stages} <= {"stagnated", "no_improvement"}
    assert history.stop_reason == "useless_mode"
