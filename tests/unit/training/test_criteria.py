import pytest

from neurom.training import (
    AnyOf,
    FixedIterations,
    MaxStages,
    RelativeChange,
    RelativeGain,
    StageRecord,
    Stagnation,
    TrainingHistory,
    Usefulness,
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


# --- Stagnation ---------------------------------------------------------


def test_stagnation_needs_more_than_patience_flat_iterations():
    criterion = Stagnation(tol=1e-4, patience=2, no_improvement=None)
    assert criterion.stop_reason([5.0, 4.0, 4.0, 4.0]) is None  # 2 flat in a row
    assert criterion.stop_reason([5.0, 4.0, 4.0, 4.0, 4.0]) == "stagnated"  # 3


def test_stagnation_count_restarts_after_a_move():
    criterion = Stagnation(tol=1e-4, patience=2, no_improvement=None)
    assert criterion.stop_reason([4.0, 4.0, 4.0, 3.0, 3.0]) is None


def test_stagnation_uses_the_relative_decrease_of_the_paper():
    # |d| = 2 |L_{k-1} - L_k| / |L_{k-1} + L_k| = 2 * 2 / 2e4 = 2e-4 between 1e4 +- 1.
    flat = [1e4 + 1, 1e4 - 1] * 2
    assert Stagnation(tol=3e-4, patience=2, no_improvement=None).stop_reason(flat) == "stagnated"
    assert Stagnation(tol=1e-4, patience=2, no_improvement=None).stop_reason(flat) is None


def test_stagnation_without_a_new_minimum():
    criterion = Stagnation(tol=0.0, patience=0, no_improvement=3)  # stagnation never fires
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0]) is None  # too short to tell
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0, 2.0]) == "no_improvement"
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0, 0.5]) is None


def test_stagnation_max_iter_wins():
    assert Stagnation(max_iter=3).stop_reason([3.0, 2.0, 1.0]) == "max_iter"


# --- FixedIterations ----------------------------------------------------


def test_fixed_iterations():
    criterion = FixedIterations(3)
    assert criterion.stop_reason([1.0, 1.0]) is None
    assert criterion.stop_reason([1.0, 1.0, 1.0]) == "n_iter"


# --- Usefulness ---------------------------------------------------------


def _records(*losses):
    return [StageRecord(stage=k, losses=list(values)) for k, values in enumerate(losses)]


def test_usefulness_never_judges_the_first_stage():
    assert Usefulness(tol=1e-4, min_useful=3).stop_reason(_records([1.0, 1.0])) is None


def test_usefulness_keeps_a_mode_that_lowered_the_loss_long_enough():
    stages = _records([1.0], [1.0, 0.9, 0.81, 0.729])  # 3 decreases in a row
    assert Usefulness(tol=1e-4, min_useful=3).stop_reason(stages) is None


def test_usefulness_count_restarts_on_stagnation():
    stages = _records([1.0], [1.0, 0.9, 0.81, 0.81, 0.7])  # 2, flat, 1
    assert Usefulness(tol=1e-4, min_useful=3).stop_reason(stages) == "useless_mode"


def test_usefulness_ignores_an_increase():
    stages = _records([1.0], [1.0, 0.9, 1.0, 0.9, 0.8])  # +, up, +, +
    assert Usefulness(tol=1e-4, min_useful=3).stop_reason(stages) is None


# --- MaxStages ----------------------------------------------------------


def test_max_stages():
    criterion = MaxStages(2)
    assert criterion.stop_reason([StageRecord(0)]) is None
    assert criterion.stop_reason([StageRecord(0), StageRecord(1)]) == "n_stages"


# --- RelativeGain -------------------------------------------------------


def _stages(*energies):
    return [StageRecord(k, final_energy=e) for k, e in enumerate(energies)]


def test_relative_gain_needs_two_stages():
    assert RelativeGain(tol=1.0).stop_reason(_stages(-1.0)) is None


def test_relative_gain_converged_below_tol():
    # gain 1e-3 relative to |E| = 100
    assert RelativeGain(tol=1e-4).stop_reason(_stages(-100.0, -100.01)) == "converged"


def test_relative_gain_continues_above_tol():
    assert RelativeGain(tol=1e-4).stop_reason(_stages(-100.0, -101.0)) is None


def test_relative_gain_stops_when_energy_rises():
    assert RelativeGain(tol=1e-4).stop_reason(_stages(-100.0, -99.0)) == "converged"


def test_relative_gain_floor_bounds_the_denominator():
    # energies near zero: the gain 1e-3 is measured against floor = 1
    assert RelativeGain(tol=1e-2, floor=1.0).stop_reason(_stages(0.0, -1e-3)) == "converged"
    assert RelativeGain(tol=1e-4, floor=1.0).stop_reason(_stages(0.0, -1e-3)) is None


# --- AnyOf --------------------------------------------------------------


def test_any_of_continues_when_none_stops():
    criterion = AnyOf(RelativeGain(tol=1e-4), MaxStages(5))
    assert criterion.stop_reason(_stages(-1.0, -2.0)) is None


def test_any_of_returns_the_first_stop_reason():
    criterion = AnyOf(RelativeGain(tol=1e-4), MaxStages(2))
    assert criterion.stop_reason(_stages(-1.0, -2.0)) == "n_stages"
    assert criterion.stop_reason(_stages(-1.0, -1.0)) == "converged"


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
