"""GreedyTrainer on the real 5-parametric beam problem.

Uses a deliberately tiny mesh and short stages: this is about the greedy
machinery driving a real CPPGD, not about solution accuracy. Accuracy is
checked in test_greedy_trainer_recovers_the_analytical_beam (Task 5).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from neurom.training import FixedIterations, GreedyTrainer, MaxStages, StageRecord

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "1d_5-parametric_beam_PGD"
    / "1d_5-parametric_beam_deflection_PGD.py"
)

TINY = {"space": 5, "E1": 4, "E2": 6, "alpha": 7, "n": 3}


def load_module():
    # The example's filename starts with a digit, so it is not importable by
    # name; load it by path.
    spec = importlib.util.spec_from_file_location("beam5p", EXAMPLE)
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


def make_trainer(problem, **kwargs):
    kwargs.setdefault("stage_criterion", FixedIterations(15))
    kwargs.setdefault("enrichment_criterion", MaxStages(3))
    return GreedyTrainer(problem.model, **kwargs)


def test_greedy_adds_one_mode_per_stage(problem):
    history = make_trainer(problem).enrich()

    assert len(history.stages) == 3
    assert problem.pgd.n_modes_truncated == 3


def test_stage_zero_trains_the_initially_active_mode_without_enriching(problem):
    # n_modes_ini modes are already active; stage 0 must train them rather than
    # immediately add a fourth.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()

    assert problem.pgd.n_modes_truncated == 1


def test_converged_energy_never_rises_when_a_mode_is_added(problem):
    # The core PGD property, and it holds between CONVERGED states only.
    #
    # It is NOT a per-iteration property here. Unlike the zero-seeded stub in
    # tests/unit/training/test_base.py, a real PGD mode is seeded at
    # Axis.init_values (0.5*ones), so it contributes from the instant it is
    # activated and the energy genuinely jumps UP at the seam; the stage then
    # has to work that back off. Hence the comparison is end-of-stage to
    # end-of-stage, via StageRecord.energy, and the stages need enough
    # iterations to actually get there.
    history = make_trainer(problem, stage_criterion=FixedIterations(80)).enrich()

    for previous, current in zip(history.stages, history.stages[1:]):
        assert current.energy <= previous.energy + 1e-9 * abs(previous.energy)


def test_frozen_modes_are_bitwise_unchanged_by_later_stages(problem):
    # The defining property of purely greedy enrichment.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()
    frozen = [field.values_reduced.detach().clone() for field in problem.pgd.monoms[0]]

    trainer.enrichment_criterion = MaxStages(3)
    trainer.enrich()

    for before, field in zip(frozen, problem.pgd.monoms[0]):
        assert torch.equal(before, field.values_reduced.detach())
    assert problem.pgd.n_modes_truncated == 3  # the second enrich did enrich


def test_capacity_stops_the_run_before_add_mode_can_raise(problem):
    # add_mode() raises at n_modes_max; should_add_stage must catch it first.
    history = make_trainer(problem, enrichment_criterion=MaxStages(99)).enrich()

    assert len(history.stages) == problem.pgd.n_modes_max
    assert history.stop_reason == "capacity"


def test_every_stage_records_duplication_diagnostics(problem):
    history = make_trainer(problem).enrich()

    for record in history.stages:
        assert record.diagnostics["amplitude"] > 0.0
        assert 0.0 <= record.diagnostics["max_correlation"] <= 1.0 + 1e-6

    # The first mode has nothing to be correlated with.
    assert history.stages[0].diagnostics["max_correlation"] == 0.0


def test_losses_stay_finite_throughout(problem):
    history = make_trainer(problem).enrich()

    assert len(history.losses) > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in history.losses)


def test_amplitude_is_the_frobenius_norm_of_the_mode(problem):
    # The existing range assertions are Cauchy-Schwarz tautologies and cannot
    # fail, so pin the actual value against an independent computation.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()

    expected = 1.0
    for field in problem.pgd.monoms[0]:
        expected *= float(field.full_values().detach().norm())

    record = trainer.history.stages[0]
    assert record.diagnostics["amplitude"] == pytest.approx(expected, rel=1e-9)


def test_max_correlation_is_one_for_a_deliberately_duplicated_mode(problem):
    # The whole point of the diagnostic: a greedy step that rediscovers a mode
    # it already has must be visible. Copy mode 0 into mode 1 verbatim and the
    # detector must report exactly 1.
    trainer = make_trainer(problem, enrichment_criterion=MaxStages(1))
    trainer.enrich()

    problem.pgd.add_mode()
    with torch.no_grad():
        for source, target in zip(problem.pgd.monoms[0], problem.pgd.monoms[1]):
            target.values_reduced.copy_(source.values_reduced)

    record = StageRecord(stage=1)
    trainer.on_stage_end(record)

    assert record.diagnostics["max_correlation"] == pytest.approx(1.0, rel=1e-6)


EXAMPLE_2P = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "1d_2-parametric_beam_PGD"
    / "1d_beam_deflection_PGD.py"
)


def load_module_2p():
    spec = importlib.util.spec_from_file_location("beam2p", EXAMPLE_2P)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_l2_error(problem):
    """Relative L2 error of the trained PGD against the analytical deflection.

    Evaluated on a dense (x, E) grid, flattened to the pointwise pairs the
    decomposition's eval-mode forward expects.
    """
    problem.model.eval()
    x = torch.linspace(problem.x_min, problem.x_max, 60)
    energy_axis = torch.linspace(problem.E_min, problem.E_max, 40)
    grid_x, grid_E = torch.meshgrid(x, energy_axis, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_E.reshape(-1)], dim=1)

    with torch.no_grad():
        predicted = problem.model(coords).reshape(-1)
    exact = (
        0.5
        * problem.load_value
        * (coords[:, 0] - problem.x_min)
        * (coords[:, 0] - problem.x_max)
        / coords[:, 1]
    )
    problem.model.train()
    return float((predicted - exact).norm() / exact.norm())


def test_greedy_trainer_recovers_the_analytical_beam():
    # The 2-parametric beam is exactly rank-1, so greedy PGD should reproduce it.
    # This is the only assertion in this file about the ANSWER rather than the
    # machinery.
    torch.manual_seed(0)
    beam2p = load_module_2p()
    problem = beam2p.build_problem(beam2p.energy)

    trainer = GreedyTrainer(
        problem.model,
        stage_criterion=beam2p.DEFAULT_STAGE_CRITERION(),
        enrichment_criterion=beam2p.DEFAULT_ENRICHMENT_CRITERION(),
    )
    trainer.enrich()

    assert relative_l2_error(problem) < beam2p.ANALYTICAL_ERROR_TOL
