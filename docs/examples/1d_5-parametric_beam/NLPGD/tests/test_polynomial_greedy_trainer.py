"""The polynomial greedy schedules on the real 5-parametric beam problem.

Same tiny mesh and short stages as ``test_greedy_trainer.py``: this is about the
schedules driving a real ``PolynomialNLPGD``, not about accuracy.

The one thing every test here ultimately guards is that the coefficient rows
actually get *released*, and that a released row is *re-frozen* when the next
mode arrives. A ``PolynomialNLPGD`` driven by a trainer that never calls
``unfreeze_mode_coefficients`` trains perfectly happily and produces a result
bit-identical to CP-PGD -- no error, no warning, just a null experiment. A
trainer that forgets to re-freeze produces something that is not greedy while
still calling itself greedy, which is worse.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from neurom.training import FixedIterations, GreedyTrainer, MaxStages, RelativeGain

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE = EXAMPLE_DIR / "1d_5-parametric_beam_deflection_NLPGD.py"

TINY = {"space": 5, "E1": 4, "E2": 6, "alpha": 7, "n": 3}

#: The two schedules the example exists to compare, plus the intermediate.
NL_SCHEDULES = ["joint", "staged", "refine"]


def load_module():
    # The example's filename starts with a digit, so it is not importable by
    # name; load it by path.
    spec = importlib.util.spec_from_file_location("beam5nl", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def beam5nl():
    return load_module()


@pytest.fixture
def problem(beam5nl):
    torch.manual_seed(0)
    return beam5nl.build_problem(beam5nl.energy, n_modes_max=3, n_nodes=TINY)


def make_trainer(beam5nl, problem, schedule="joint", **kwargs):
    kwargs.setdefault("stage_criterion", FixedIterations(10))
    # MaxStages counts MODES here, not stages -- should_add_stage feeds the
    # criterion the end-of-mode records only. Two modes is enough to exercise
    # the add_mode seam while keeping the test seconds long.
    kwargs.setdefault("enrichment_criterion", MaxStages(2))
    return beam5nl.STRATEGIES[schedule](problem.model, **kwargs)


# --- the schedules' shapes -------------------------------------------------


@pytest.mark.parametrize(
    "schedule, kinds",
    [
        ("joint", ["joint"]),
        ("staged", ["cp", "corr"]),
        ("refine", ["cp", "joint"]),
    ],
)
def test_each_schedule_runs_its_declared_stages_per_mode(
    beam5nl, problem, schedule, kinds
):
    history = make_trainer(beam5nl, problem, schedule).enrich()

    assert problem.pgd.n_modes_truncated == 2
    assert len(history.stages) == 2 * len(kinds)
    assert [r.diagnostics["kind"] for r in history.stages] == kinds * 2


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_max_stages_counts_modes_not_stages(beam5nl, problem, schedule):
    # What makes the schedules comparable at all: MaxStages(2) must mean rank 2
    # under every one of them, or a two-stage schedule would silently be scored
    # at half the rank of a one-stage one.
    make_trainer(beam5nl, problem, schedule).enrich()

    assert problem.pgd.n_modes_truncated == 2


# --- what each stage is allowed to move ------------------------------------


@pytest.mark.parametrize("schedule", ["staged", "refine"])
def test_the_cp_stage_leaves_the_coefficients_frozen_at_zero(
    beam5nl, problem, schedule
):
    # The half of a two-stage schedule that must behave exactly like CP-PGD.
    # Checked at the stage boundary rather than after the run, because after the
    # run every row has been released and the distinction is gone.
    trainer = make_trainer(beam5nl, problem, schedule)
    trainer.prepare_stage(0)
    trainer.stage(0)

    row = problem.pgd.coefficients[0]
    assert not row.requires_grad
    assert torch.equal(row, torch.zeros_like(row))


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_every_schedule_ends_a_mode_with_a_fitted_coefficient_row(
    beam5nl, problem, schedule
):
    # The reason these trainers exist. Whatever the schedule, a finished mode
    # must carry a non-zero, finite correction.
    trainer = make_trainer(beam5nl, problem, schedule)
    for stage in range(trainer.stages_per_mode):
        trainer.prepare_stage(stage)
        trainer.stage(stage)

    row = problem.pgd.coefficients[0]
    assert row.requires_grad
    assert row.abs().max() > 0.0
    assert torch.isfinite(row).all()


def test_joint_moves_the_monoms_and_the_coefficients_in_the_same_stage(
    beam5nl, problem
):
    # The defining property of strategy A: there is no CP-only phase at all.
    trainer = make_trainer(beam5nl, problem, "joint")
    trainer.prepare_stage(0)
    before = [f.values_reduced.detach().clone() for f in problem.pgd.monoms[0]]

    assert problem.pgd.coefficients[0].requires_grad
    for field in problem.pgd.monoms[0]:
        assert field.values_reduced.requires_grad
    trainer.stage(0)

    assert problem.pgd.coefficients[0].abs().max() > 0.0
    assert any(
        not torch.equal(b, f.values_reduced.detach())
        for b, f in zip(before, problem.pgd.monoms[0])
    )


def test_staged_freezes_the_monoms_while_fitting_the_coefficients(beam5nl, problem):
    # The defining property of strategy B.
    trainer = make_trainer(beam5nl, problem, "staged")
    trainer.prepare_stage(0)
    trainer.stage(0)

    trainer.prepare_stage(1)
    for field in problem.pgd.monoms[0]:
        assert not field.values_reduced.requires_grad
    # Snapshot AFTER prepare_stage, not before: prepare_stage also runs
    # fix_gauge(), which rescales the monoms (exactly, field-preservingly) even
    # though they are frozen. What the schedule promises is that the *optimizer*
    # does not touch them, which is what the loop below measures.
    monoms = [f.values_reduced.detach().clone() for f in problem.pgd.monoms[0]]
    trainer.stage(1)

    for before, field in zip(monoms, problem.pgd.monoms[0]):
        assert torch.equal(before, field.values_reduced.detach())
    assert problem.pgd.coefficients[0].abs().max() > 0.0


def test_refine_unfreezes_the_monoms_alongside_the_coefficients(beam5nl, problem):
    # The one difference between "refine" and "staged", stated as a test so the
    # two cannot silently collapse into each other.
    trainer = make_trainer(beam5nl, problem, "refine")
    trainer.prepare_stage(0)
    trainer.stage(0)

    trainer.prepare_stage(1)

    assert problem.pgd.coefficients[0].requires_grad
    for field in problem.pgd.monoms[0]:
        assert field.values_reduced.requires_grad


# --- the greedy guarantee, extended to the coefficients --------------------


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_a_finished_modes_coefficients_are_refrozen_by_the_next_mode(
    beam5nl, problem, schedule
):
    # "Purely progressive" has to cover the coefficients too, or the schedule
    # would be greedy in the monoms and simultaneous in C -- which is neither of
    # the two strategies under comparison. freeze_all() is what delivers this;
    # this test is what stops it being quietly dropped.
    trainer = make_trainer(beam5nl, problem, schedule)
    spm = trainer.stages_per_mode
    for stage in range(spm):  # mode 0, complete
        trainer.prepare_stage(stage)
        trainer.stage(stage)
    assert problem.pgd.coefficients[0].requires_grad  # released, mode still open

    trainer.prepare_stage(spm)  # mode 1 arrives

    assert not problem.pgd.coefficients[0].requires_grad
    for field in problem.pgd.monoms[0]:
        assert not field.values_reduced.requires_grad


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_a_finished_modes_coefficients_never_move_again(beam5nl, problem, schedule):
    # The value-level counterpart of the test above. Not bitwise: the gauge fix
    # rescales C along with the monoms at every stage boundary, exactly and
    # field-preservingly (see test_greedy_trainer). renormalise=False is what
    # isolates "the optimizer never touches it" from "nothing ever touches it".
    #
    # Driven through enrich(), not prepare_stage/stage by hand: enrich() picks
    # its next stage index off len(history.stages), so hand-run stages that
    # never enter the history would make it restart at mode 0 and retrain it.
    trainer = make_trainer(
        beam5nl, problem, schedule,
        enrichment_criterion=MaxStages(1), renormalise=False,
    )
    trainer.enrich()
    assert problem.pgd.n_modes_truncated == 1
    finished = problem.pgd.coefficients[0].detach().clone()
    assert finished.abs().max() > 0.0  # there is something to keep fixed

    trainer.enrichment_criterion = MaxStages(3)
    trainer.enrich()

    assert problem.pgd.n_modes_truncated == 3  # the run did enrich
    assert torch.equal(finished, problem.pgd.coefficients[0].detach())


# --- bookkeeping and the baseline ------------------------------------------


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_a_completed_mode_does_not_raise_the_energy(beam5nl, problem, schedule):
    # C = 0 is in the feasible set of every stage that releases a row, so a
    # completed mode's energy is at worst its CP stage's. A descent method can
    # overshoot, hence the tolerance; but a mode that ENDS above where its first
    # stage left it means the release wired something wrong -- a sign error, or
    # coefficients entering the energy but not its gradient.
    trainer = make_trainer(beam5nl, problem, schedule)
    history = trainer.enrich()
    spm = trainer.stages_per_mode
    if spm == 1:
        pytest.skip("one stage per mode: there is no within-mode pair to compare")

    for first, last in zip(history.stages[::spm], history.stages[spm - 1 :: spm]):
        assert last.energy <= first.energy + 1e-6 * abs(first.energy)


@pytest.mark.parametrize("schedule", NL_SCHEDULES)
def test_enrichment_gains_are_measured_mode_to_mode(beam5nl, problem, schedule):
    # The bookkeeping subtlety: should_add_stage feeds the enrichment criterion
    # the END-OF-MODE records only. Handed the raw list under a two-stage
    # schedule, RelativeGain would compare a correction stage against the CP
    # stage of the SAME mode and stop after one mode. Pinned with a tolerance
    # that is easily met mode-to-mode, so reaching capacity is the tell.
    trainer = make_trainer(
        beam5nl, problem, schedule, enrichment_criterion=RelativeGain(tol=1e-12)
    )
    history = trainer.enrich()

    assert problem.pgd.n_modes_truncated == 3
    assert history.stop_reason == "capacity"


def test_an_unknown_schedule_is_rejected(beam5nl, problem):
    with pytest.raises(ValueError, match="unknown schedule"):
        beam5nl.PolynomialGreedyTrainer(problem.model, schedule="nonsense")


def test_a_cp_baseline_trainer_leaves_every_coefficient_at_zero(beam5nl, problem):
    # The control the whole comparison rests on, stated as a test rather than as
    # a comment: the library's GreedyTrainer never releases a row, so its run IS
    # CP-PGD. If this ever stops holding, the "cp-baseline" ledger rows silently
    # stop being a baseline.
    GreedyTrainer(
        problem.model,
        stage_criterion=FixedIterations(10),
        enrichment_criterion=MaxStages(3),
    ).enrich()

    for row in problem.pgd.coefficients:
        assert not row.requires_grad
        assert torch.equal(row, torch.zeros_like(row))
