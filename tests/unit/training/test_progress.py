import io

import torch

from neurom.training import (
    FixedIterations,
    GreedyTrainer,
    MaxStages,
    Phase,
    ProgressLog,
    ProgressReporter,
    Stagnation,
)

torch.set_default_dtype(torch.float32)

# The toy model `make_model` (rank-2 L2 projection) is in conftest.py.


def _phase(name, n_iter):
    return Phase(lambda p: torch.optim.Adam(p, lr=0.1), FixedIterations(n_iter), name)


class Recorder(ProgressReporter):
    """Keeps every hook call, in order."""

    def __init__(self):
        self.calls = []

    def stage_start(self, stage_index):
        self.calls.append(("stage_start", stage_index))

    def phase_start(self, name, max_iter=None):
        self.calls.append(("phase_start", name))

    def update(self, iteration, loss):
        self.calls.append(("update", iteration))

    def phase_end(self, name, n_iter, reason, loss):
        self.calls.append(("phase_end", name, n_iter, reason))

    def stage_end(self, record):
        self.calls.append(("stage_end", record.stage))

    def close(self, history):
        self.calls.append(("close", len(history.stages)))


def test_hooks_are_called_in_order(make_model):
    recorder = Recorder()
    trainer = GreedyTrainer(
        make_model(),
        phases=[_phase("adam", 2), _phase("second", 1)],
        enrichment_criterion=MaxStages(2),
        progress=recorder,
    )
    trainer.enrich()

    def one_stage(m):
        return [
            ("stage_start", m),
            ("phase_start", "adam"), ("update", 1), ("update", 2), ("phase_end", "adam", 2, "n_iter"),
            ("phase_start", "second"), ("update", 1), ("phase_end", "second", 1, "n_iter"),
            ("stage_end", m),
        ]

    assert recorder.calls == one_stage(0) + one_stage(1) + [("close", 2)]


def test_update_count_matches_the_record(make_model):
    recorder = Recorder()
    history = GreedyTrainer(
        make_model(), phases=[_phase("adam", 4)], enrichment_criterion=MaxStages(3),
        progress=recorder,
    ).enrich()
    n_updates = sum(1 for call in recorder.calls if call[0] == "update")
    assert n_updates == sum(record.n_iter for record in history.stages)


def test_log_writes_one_line_per_event(make_model):
    stream = io.StringIO()
    GreedyTrainer(
        make_model(), phases=[_phase("adam", 4)], enrichment_criterion=MaxStages(2),
        progress=ProgressLog(every=2, stream=stream),
    ).enrich()
    lines = stream.getvalue().splitlines()
    assert lines[0] == "mode 0: start"
    assert [line.split()[2] for line in lines if line.startswith("  adam   it")] == ["2", "4", "2", "4"]
    assert sum(line.startswith("  adam   done: 4 it, n_iter") for line in lines) == 2
    assert lines[-2].startswith("mode 1: done in 4 it, n_iter, energy =")
    assert lines[-1].startswith("run done: 2 modes, n_stages,")


def test_silent_by_default(make_model, capsys):
    GreedyTrainer(make_model(), phases=[_phase("adam", 3)], enrichment_criterion=MaxStages(2)).enrich()
    assert capsys.readouterr().out == ""


def test_log_reports_the_final_phases(make_model):
    stream = io.StringIO()
    GreedyTrainer(
        make_model(), phases=[_phase("adam", 2)], enrichment_criterion=MaxStages(1),
        final_phases=[_phase("polish", 3)], progress=ProgressLog(every=None, stream=stream),
    ).enrich()
    lines = stream.getvalue().splitlines()
    assert lines[-4] == "final: start"
    assert lines[-3].startswith("  polish done: 3 it, n_iter")
    assert lines[-2].startswith("final: done in 3 it, n_iter, energy =")
    assert lines[-1].startswith("run done: 1 modes, n_stages,")


def _logged_iterations(stream, name="adam"):
    return [int(line.split()[2]) for line in stream.getvalue().splitlines()
            if line.startswith(f"  {name:<6} it")]


def test_log_every_fraction_of_max_iter(make_model):
    stream = io.StringIO()
    criterion = Stagnation(tol=0.0, patience=0, no_improvement=None, max_iter=20)
    GreedyTrainer(
        make_model(), phases=[Phase(lambda p: torch.optim.Adam(p, lr=0.1), criterion, "adam")],
        enrichment_criterion=MaxStages(1), progress=ProgressLog(every=0.1, stream=stream),
    ).enrich()
    assert _logged_iterations(stream) == list(range(2, 21, 2))


def test_log_every_fraction_without_max_iter_prints_no_iteration(make_model):
    stream = io.StringIO()
    GreedyTrainer(
        make_model(), phases=[_phase("adam", 4)], enrichment_criterion=MaxStages(1),
        progress=ProgressLog(every=0.5, stream=stream),
    ).enrich()
    assert _logged_iterations(stream) == []


def test_log_every_per_phase(make_model):
    stream = io.StringIO()
    GreedyTrainer(
        make_model(), phases=[_phase("adam", 4), _phase("second", 3), _phase("quiet", 2)],
        enrichment_criterion=MaxStages(1),
        progress=ProgressLog(every={"adam": 2, "second": 1}, stream=stream),
    ).enrich()
    assert _logged_iterations(stream) == [2, 4]
    assert _logged_iterations(stream, "second") == [1, 2, 3]
    assert _logged_iterations(stream, "quiet") == []
