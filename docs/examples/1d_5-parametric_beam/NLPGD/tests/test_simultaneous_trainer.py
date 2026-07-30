"""SimultaneousTrainer on the real 5-parametric beam problem.

Same tiny mesh and short stages as ``test_greedy_trainer.py``: this is about the
strategy driving a real CPPGD, not about solution accuracy. Accuracy against the
saved FEM reference is what ``main(trainer_cls=SimultaneousTrainer)`` reports and
what the CHANGELOG records; it needs full-size meshes and 300-iteration stages,
far more than a test should spend.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from neurom.training import (
    FixedIterations,
    GreedyTrainer,
    MaxStages,
    SimultaneousTrainer,
    StageRecord,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE = EXAMPLE_DIR / "1d_5-parametric_beam_deflection_NLPGD.py"

TINY = {"space": 5, "E1": 4, "E2": 6, "alpha": 7, "n": 3}


def load_module():
    # The example's filename starts with a digit, so it is not importable by
    # name; load it by path.
    spec = importlib.util.spec_from_file_location("beam5nl", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def beam5p():
    return load_module()


@pytest.fixture
def problem(beam5p):
    torch.manual_seed(0)
    return beam5p.build_problem(beam5p.energy, n_modes_max=3, n_nodes=TINY)


def make_trainer(problem, cls=SimultaneousTrainer, **kwargs):
    kwargs.setdefault("stage_criterion", FixedIterations(15))
    kwargs.setdefault("enrichment_criterion", MaxStages(3))
    return cls(problem.model, **kwargs)


def mode_values(problem, mode):
    return [field.values_reduced.detach().clone() for field in problem.pgd.monoms[mode]]


def test_one_mode_is_added_per_stage(problem):
    history = make_trainer(problem).enrich()

    assert len(history.stages) == 3
    assert problem.pgd.n_modes_truncated == 3


def test_stage_zero_trains_the_initially_active_mode_without_enriching(problem):
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()

    assert problem.pgd.n_modes_truncated == 1


def test_earlier_modes_keep_moving_in_later_stages(problem):
    # The defining property, and the exact mirror of the greedy trainer's
    # test_frozen_modes_are_bitwise_unchanged_by_later_stages.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()
    after_first_stage = mode_values(problem, 0)

    trainer.enrichment_criterion = MaxStages(3)
    trainer.enrich()

    assert problem.pgd.n_modes_truncated == 3
    assert not any(
        torch.equal(before, after)
        for before, after in zip(after_first_stage, mode_values(problem, 0))
    )


def test_every_active_mode_is_trainable_and_no_inactive_one_is(problem):
    trainer = make_trainer(problem)
    trainer.prepare_stage(0)
    trainer.prepare_stage(1)

    for mode in range(2):
        for field in problem.pgd.monoms[mode]:
            assert field.values_reduced.requires_grad
    for field in problem.pgd.monoms[2]:
        assert not field.values_reduced.requires_grad


def test_a_decomposition_handed_over_frozen_is_unfrozen(problem):
    # add_mode() only unfreezes the mode it adds, so a decomposition arriving
    # frozen -- from a greedy run, say -- would otherwise train one mode and
    # call it simultaneous. This is what makes the two strategies composable.
    greedy = make_trainer(problem, cls=GreedyTrainer, enrichment_criterion=MaxStages(2))
    greedy.enrich()
    frozen = mode_values(problem, 0)

    simultaneous = make_trainer(problem, enrichment_criterion=MaxStages(3))
    simultaneous.history = greedy.history
    simultaneous.enrich()

    assert not any(
        torch.equal(before, after)
        for before, after in zip(frozen, mode_values(problem, 0))
    )


def test_capacity_stops_the_run_before_add_mode_can_raise(problem):
    history = make_trainer(problem, enrichment_criterion=MaxStages(99)).enrich()

    assert len(history.stages) == problem.pgd.n_modes_max
    assert history.stop_reason == "capacity"


def test_losses_stay_finite_throughout(problem):
    history = make_trainer(problem).enrich()

    assert len(history.losses) > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in history.losses)


def test_diagnostics_see_a_duplicate_pair_that_excludes_the_newest_mode(problem):
    # The difference from the greedy diagnostic. With nothing frozen, two
    # earlier modes can collapse onto each other while the newest one is fine;
    # a newest-mode-only measure reports that state as clean.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(2))
    trainer.enrich()
    problem.pgd.add_mode()

    with torch.no_grad():
        for source, target in zip(problem.pgd.monoms[0], problem.pgd.monoms[1]):
            target.values_reduced.copy_(source.values_reduced)

        # Make mode 2 orthogonal to the duplicated pair on the space axis. The
        # constrained DOFs are homogeneous (zero) on every mode, so projecting
        # in reduced space is projecting in full space -- which is what the
        # diagnostic reads. Without this step the newest mode is still at its
        # 0.5-constant seed and correlates 0.99997 with the trained mode 0, so
        # the contrast below would say nothing.
        earlier = problem.pgd.monoms[0][0].values_reduced.reshape(-1)
        newest = problem.pgd.monoms[2][0].values_reduced
        newest -= (earlier @ newest.reshape(-1) / (earlier @ earlier)) * (
            earlier.reshape(newest.shape)
        )

    record = StageRecord(stage=2)
    trainer.on_stage_end(record)

    assert record.diagnostics["max_correlation"] == pytest.approx(1.0, rel=1e-6)
    # ... and the greedy diagnostic, looking only at how mode 2 compares to its
    # predecessors, calls the very same state clean.
    greedy_record = StageRecord(stage=2)
    GreedyTrainer(problem.model).on_stage_end(greedy_record)
    # 1e-3, not tighter: the projection above is done in float32, and its
    # residual leaves ~1e-4 of correlation behind. Four orders of magnitude
    # below the 1.0 the pairwise measure reports is contrast enough.
    assert greedy_record.diagnostics["max_correlation"] == pytest.approx(0.0, abs=1e-3)


def test_amplitude_is_the_frobenius_norm_of_the_newest_mode(problem):
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(2))
    trainer.enrich()

    expected = 1.0
    for field in problem.pgd.monoms[1]:
        expected *= float(field.full_values().detach().norm())

    assert trainer.history.stages[-1].diagnostics["amplitude"] == pytest.approx(
        expected, rel=1e-9
    )
