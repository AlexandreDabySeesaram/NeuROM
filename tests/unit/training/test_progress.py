"""Unit tests for progress reporting.

The bar writes to an injected stream, so what it draws is testable without a
terminal. What matters is that the loop calls the reporter at the right moments
and that the default reporter is silent.
"""

import io

import torch

from neurom.training import FixedIterations, MaxStages, ProgressBar, ProgressReporter
from neurom.training.criteria import StageCriterion
from tests.unit.training.test_base import StubModel, StubTrainer


class RecordingProgress(ProgressReporter):
    def __init__(self):
        self.calls = []

    def stage_start(self, stage_index, budget=None):
        self.calls.append(("start", stage_index, budget))

    def update(self, iteration, loss):
        self.calls.append(("update", iteration))

    def stage_end(self, record):
        self.calls.append(("end", record.stage, record.stop_reason))

    def close(self):
        self.calls.append(("close",))


def make_trainer(progress=None, n_iter=5, n_stages=2):
    torch.manual_seed(0)
    return StubTrainer(
        StubModel(),
        stage_criterion=FixedIterations(n_iter),
        enrichment_criterion=MaxStages(n_stages),
        progress=progress,
    )


def test_the_loop_is_silent_by_default(capsys):
    # Training nobody is watching prints nothing -- tests and batch scripts get
    # a clean stdout AND stderr without asking for it.
    make_trainer().enrich()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_the_reporter_sees_every_stage_and_every_iteration():
    progress = RecordingProgress()
    make_trainer(progress, n_iter=5, n_stages=2).enrich()

    assert progress.calls[0] == ("start", 0, 5)
    assert [call for call in progress.calls if call[0] == "update"] == [
        ("update", i) for i in range(1, 6)
    ] * 2
    assert ("end", 0, "n_iter") in progress.calls
    assert progress.calls[-1] == ("close",)


def test_a_diverged_stage_still_reports_its_end():
    # The bar must not be left mid-line when a run blows up.
    progress = RecordingProgress()
    trainer = make_trainer(progress, n_iter=200, n_stages=2)
    trainer.optimizer_factory = lambda params: torch.optim.Adam(params, lr=1e30)
    trainer.enrich()

    ends = [call for call in progress.calls if call[0] == "end"]
    assert ends[-1][2] == "diverged"
    assert progress.calls[-1] == ("close",)


def test_the_budget_comes_from_the_stage_criterion():
    class Unbounded(StageCriterion):
        def stop_reason(self, losses):
            return "done" if len(losses) >= 3 else None

    progress = RecordingProgress()
    trainer = make_trainer(progress, n_stages=1)
    trainer.stage_criterion = Unbounded()
    trainer.enrich()

    # A criterion with no cap says None rather than guessing, and the bar
    # degrades to a counter instead of showing a wrong fraction.
    assert progress.calls[0] == ("start", 0, None)


def test_the_bar_draws_a_fraction_when_the_budget_is_known():
    stream = io.StringIO()
    bar = ProgressBar(stream=stream, width=10, every=0.0)
    make_trainer(bar, n_iter=4, n_stages=1).enrich()
    drawn = stream.getvalue()

    assert "stage 0" in drawn
    assert "4/4" in drawn  # the last in-loop draw, at full budget
    assert "n_iter" in drawn  # the finalised line carries the stop reason
    assert drawn.endswith("\n")
    assert drawn.count("\r") >= 4  # redrawn in place, not one line per iteration


def test_the_bar_degrades_to_a_counter_without_a_budget():
    stream = io.StringIO()
    bar = ProgressBar(stream=stream, width=10, every=0.0)
    bar.stage_start(2, budget=None)
    bar.update(7, -1.5)

    drawn = stream.getvalue()
    assert "|" not in drawn
    assert "7" in drawn


def test_a_short_stage_still_draws_something():
    # The redraw throttle must not swallow a stage that ends inside its window:
    # a run that prints nothing at all looks hung.
    stream = io.StringIO()
    bar = ProgressBar(stream=stream, width=10, every=3600.0)
    bar.stage_start(0, budget=100)
    bar.update(1, -1.0)

    assert "stage 0" in stream.getvalue()
