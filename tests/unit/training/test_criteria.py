import pytest

from neurom.training import (
    FixedIterations,
    MaxStages,
    RelativeChange,
    StageRecord,
    TrainingHistory,
)


# --- RelativeChange -----------------------------------------------------


def test_relative_change_needs_two_losses():
    assert RelativeChange(tol=1.0).stop_reason([1.0]) is None


def test_relative_change_converged_below_tol():
    assert RelativeChange(tol=1e-3).stop_reason([1.0, 1.0005]) == "converged"


def test_relative_change_continues_above_tol():
    assert RelativeChange(tol=1e-3).stop_reason([1.0, 1.01]) is None


def test_relative_change_is_relative_to_current_loss():
    # |dL| = 1, |L| = 1e4: relative change 1e-4.
    assert RelativeChange(tol=2e-4).stop_reason([1e4 + 1, 1e4]) == "converged"
    assert RelativeChange(tol=5e-5).stop_reason([1e4 + 1, 1e4]) is None


def test_relative_change_zero_loss_does_not_divide_by_zero():
    assert RelativeChange(tol=1e-6).stop_reason([0.0, 0.0]) == "converged"


def test_relative_change_max_iter_wins():
    assert RelativeChange(tol=0.0, max_iter=3).stop_reason([3.0, 2.0, 1.0]) == "max_iter"


# --- FixedIterations ----------------------------------------------------


def test_fixed_iterations():
    criterion = FixedIterations(3)
    assert criterion.stop_reason([1.0, 1.0]) is None
    assert criterion.stop_reason([1.0, 1.0, 1.0]) == "n_iter"


# --- MaxStages ----------------------------------------------------------


def test_max_stages():
    criterion = MaxStages(2)
    assert criterion.stop_reason([StageRecord(0)]) is None
    assert criterion.stop_reason([StageRecord(0), StageRecord(1)]) == "n_stages"


# --- history ------------------------------------------------------------


def test_stage_record_energy_prefers_final_energy():
    assert StageRecord(0, losses=[3.0, 2.0], final_energy=1.0).energy == pytest.approx(1.0, rel=1e-12)
    assert StageRecord(0, losses=[3.0, 2.0]).energy == pytest.approx(2.0, rel=1e-12)
    assert StageRecord(0).n_iter == 0


def test_training_history_concatenates_losses():
    history = TrainingHistory()
    history.append(StageRecord(0, losses=[3.0, 2.0]))
    history.append(StageRecord(1, losses=[1.0]))
    assert history.losses == [3.0, 2.0, 1.0]
