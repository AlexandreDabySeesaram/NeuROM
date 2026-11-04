import pytest

from neurom.training import (
    AnyOf,
    FixedIterations,
    MaxStages,
    RelativeGain,
    StageRecord,
    Stagnation,
    TrainingHistory,
    Usefulness,
)


# --- Stagnation ---------------------------------------------------------


def test_stagnation_without_patience_stops_at_the_first_flat_iteration():
    criterion = Stagnation(tol=1e-3, patience=0, no_improvement=None)
    assert criterion.stop_reason([1.0]) is None
    assert criterion.stop_reason([1.0, 1.01]) is None
    assert criterion.stop_reason([1.0, 1.0005]) == "stagnated"


def test_stagnation_zero_loss_does_not_divide_by_zero():
    assert Stagnation(tol=1e-6, patience=0).stop_reason([0.0, 0.0]) == "stagnated"


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
    assert (
        Stagnation(tol=3e-4, patience=2, no_improvement=None).stop_reason(flat)
        == "stagnated"
    )
    assert (
        Stagnation(tol=1e-4, patience=2, no_improvement=None).stop_reason(flat) is None
    )


def test_stagnation_without_a_new_minimum():
    criterion = Stagnation(
        tol=0.0, patience=0, no_improvement=3
    )  # stagnation never fires
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0]) is None  # too short to tell
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0, 2.0]) == "no_improvement"
    assert criterion.stop_reason([1.0, 3.0, 2.0, 3.0, 0.5]) is None


def test_stagnation_max_iter_wins():
    assert Stagnation(max_iter=3).stop_reason([3.0, 2.0, 1.0]) == "max_iter"


def test_stagnation_tol_per_stage():
    criterion = Stagnation(
        tol=lambda m: 1e-2 if m == 0 else 1e-4, patience=0, no_improvement=None
    )
    losses = [1.0, 0.999]  # |d| ~ 1e-3
    criterion.start(0)
    assert criterion.stop_reason(losses) == "stagnated"
    criterion.start(1)
    assert criterion.stop_reason(losses) is None


def test_stagnation_tol_function_before_start_is_tol_of_stage_zero():
    criterion = Stagnation(
        tol=lambda m: 1e-2 * 10.0**-m, patience=0, no_improvement=None
    )
    assert criterion.stage_tol == pytest.approx(1e-2, rel=1e-12)
    assert criterion.stop_reason([1.0, 0.999]) == "stagnated"


def test_stagnation_constant_tol_ignores_the_stage():
    criterion = Stagnation(tol=1e-3, patience=0, no_improvement=None)
    criterion.start(7)
    assert criterion.stage_tol == pytest.approx(1e-3, rel=1e-12)


# --- FixedIterations ----------------------------------------------------


def test_fixed_iterations():
    criterion = FixedIterations(3)
    assert criterion.stop_reason([1.0, 1.0]) is None
    assert criterion.stop_reason([1.0, 1.0, 1.0]) == "n_iter"


# --- Usefulness ---------------------------------------------------------


def _records(*losses):
    return [
        StageRecord(stage=k, losses=list(values)) for k, values in enumerate(losses)
    ]


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
    assert (
        RelativeGain(tol=1e-2, floor=1.0).stop_reason(_stages(0.0, -1e-3))
        == "converged"
    )
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
    assert StageRecord(0, losses=[3.0, 2.0], final_energy=1.0).energy == pytest.approx(
        1.0, rel=1e-12
    )
    assert StageRecord(0, losses=[3.0, 2.0]).energy == pytest.approx(2.0, rel=1e-12)
    assert StageRecord(0).n_iter == 0


def test_training_history_concatenates_losses():
    history = TrainingHistory()
    history.append(StageRecord(0, losses=[3.0, 2.0]))
    history.append(StageRecord(1, losses=[1.0]))
    assert history.losses == [3.0, 2.0, 1.0]
