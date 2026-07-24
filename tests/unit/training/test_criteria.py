from neurom.training.criteria import (
    RelativeChange,
    FixedIterations,
    RelativeGain,
    MaxStages,
)
from neurom.training.history import StageRecord


class TestFixedIterations:
    def test_runs_exactly_n_iterations(self):
        criterion = FixedIterations(3)
        assert criterion.stop_reason([1.0]) is None
        assert criterion.stop_reason([1.0, 0.5]) is None
        assert criterion.stop_reason([1.0, 0.5, 0.2]) == "n_iter"


class TestRelativeChange:
    def test_does_not_stop_before_min_iter_even_on_a_flat_history(self):
        # THE bug this guards: Adam has a long sticky early phase on the beam
        # problem where the energy barely moves. A plateau detector without
        # min_iter reads that as convergence and stops mode 0 after ~20
        # iterations. See tests/integration/test_1d_beam_deflection_PGD.py.
        criterion = RelativeChange(tol=1e-2, window=2, max_iter=1000, min_iter=10)
        flat = [5.0] * 9
        assert criterion.stop_reason(flat) is None

    def test_stops_once_the_plateau_persists_past_min_iter(self):
        criterion = RelativeChange(tol=1e-2, window=2, max_iter=1000, min_iter=10)
        flat = [5.0] * 10
        assert criterion.stop_reason(flat) == "converged"

    def test_keeps_going_while_improving_fast(self):
        criterion = RelativeChange(tol=1e-2, window=2, max_iter=1000, min_iter=3)
        improving = [100.0, 80.0, 60.0, 40.0]
        assert criterion.stop_reason(improving) is None

    def test_stops_at_max_iter_with_a_distinct_reason(self):
        criterion = RelativeChange(tol=0.0, window=2, max_iter=5, min_iter=1)
        improving = [100.0, 80.0, 60.0, 40.0, 20.0]
        assert criterion.stop_reason(improving) == "max_iter"

    def test_floor_keeps_the_denominator_sane_across_the_zero_crossing(self):
        # The beam energy crosses zero (~+2e5 -> ~-1e8). Near the crossing the
        # magnitudes are tiny, so a purely relative denominator turns a
        # negligible absolute change into an enormous ratio and the stage never
        # stops. A/B on the floor with identical data proves it is the floor
        # doing the work: an absolute change of 0.002 is nothing.
        losses = [0.001, -0.001]
        floored = RelativeChange(
            tol=1e-2, window=1, max_iter=1000, min_iter=1, floor=1.0
        )
        unfloored = RelativeChange(
            tol=1e-2, window=1, max_iter=1000, min_iter=1, floor=1e-12
        )

        assert floored.stop_reason(losses) == "converged"  # 0.002 / 1.0
        assert unfloored.stop_reason(losses) is None  # 0.002 / 0.001 = 2.0

    def test_a_rising_loss_stops_the_stage(self):
        # rel_improvement is signed, so going backwards counts as no progress.
        criterion = RelativeChange(tol=1e-2, window=1, max_iter=1000, min_iter=1)
        assert criterion.stop_reason([10.0, 20.0]) == "converged"

    def test_needs_more_than_window_samples_before_judging(self):
        criterion = RelativeChange(tol=1e-2, window=5, max_iter=1000, min_iter=1)
        assert criterion.stop_reason([5.0, 5.0, 5.0]) is None


class TestRelativeGain:
    def _stage(self, index, energy):
        return StageRecord(stage=index, losses=[energy])

    def test_never_stops_before_two_stages_exist(self):
        criterion = RelativeGain(tol=1e-2)
        assert criterion.stop_reason([]) is None
        assert criterion.stop_reason([self._stage(0, 10.0)]) is None

    def test_stops_when_a_stage_barely_improved_on_the_last(self):
        criterion = RelativeGain(tol=1e-2)
        stages = [self._stage(0, 100.0), self._stage(1, 99.999)]
        assert criterion.stop_reason(stages) == "converged"

    def test_keeps_going_while_stages_still_pay(self):
        criterion = RelativeGain(tol=1e-2)
        stages = [self._stage(0, 100.0), self._stage(1, 50.0)]
        assert criterion.stop_reason(stages) is None


class TestMaxStages:
    def test_stops_once_the_stage_budget_is_spent(self):
        criterion = MaxStages(2)
        assert criterion.stop_reason([StageRecord(stage=0)]) is None
        assert (
            criterion.stop_reason([StageRecord(stage=0), StageRecord(stage=1)])
            == "n_stages"
        )
