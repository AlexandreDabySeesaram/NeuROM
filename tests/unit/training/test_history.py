import math

from neurom.training.history import StageRecord, TrainingHistory


def test_record_derives_n_iter_and_energy_from_losses():
    record = StageRecord(stage=0, losses=[5.0, 3.0, 2.0])
    assert record.n_iter == 3
    assert record.energy == 2.0


def test_empty_record_reports_nan_energy():
    record = StageRecord(stage=0)
    assert record.n_iter == 0
    assert math.isnan(record.energy)


def test_append_sets_gain_relative_to_previous_stage():
    history = TrainingHistory()
    history.append(StageRecord(stage=0, losses=[10.0, 4.0]))
    history.append(StageRecord(stage=1, losses=[4.0, 1.0]))

    assert math.isnan(history.stages[0].gain)  # no previous stage
    assert history.stages[1].gain == 3.0       # 4.0 -> 1.0


def test_losses_concatenates_every_stage_in_order():
    history = TrainingHistory()
    history.append(StageRecord(stage=0, losses=[10.0, 4.0]))
    history.append(StageRecord(stage=1, losses=[4.0, 1.0]))

    assert history.losses == [10.0, 4.0, 4.0, 1.0]


def test_records_carry_independent_mutable_defaults():
    # Guards the classic dataclass mutable-default bug: two records sharing one
    # list would make every stage's history identical.
    first, second = StageRecord(stage=0), StageRecord(stage=1)
    first.losses.append(1.0)
    first.diagnostics["amplitude"] = 2.0

    assert second.losses == []
    assert second.diagnostics == {}


def test_energy_prefers_the_recorded_end_of_stage_value():
    # losses[-1] is the loss BEFORE the last update, so a stage that recorded
    # its true final energy must report that instead.
    record = StageRecord(stage=0, losses=[5.0, 3.0, 2.0], final_energy=1.5)
    assert record.energy == 1.5


def test_energy_falls_back_to_the_last_loss_when_no_final_energy_was_recorded():
    record = StageRecord(stage=0, losses=[5.0, 3.0, 2.0])
    assert record.energy == 2.0


def test_energy_reports_a_genuinely_nan_final_energy_instead_of_falling_back():
    # A NaN sentinel could not tell "never computed" from "computed and NaN",
    # so a diverged stage reported a stale finite loss as its energy.
    record = StageRecord(stage=0, losses=[5.0, 3.0], final_energy=float("nan"))
    assert math.isnan(record.energy)
