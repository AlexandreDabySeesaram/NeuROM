"""Unit tests for checkpointing, against the stub CP decomposition.

The point of a checkpoint here is "train once, plot many times", so the tests
are about a *reloaded* model producing the same numbers as the trained one --
not about the file's contents.
"""

import pytest
import torch

from neurom.training import (
    FixedIterations,
    MaxStages,
    SimultaneousTrainer,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit.training.test_simultaneous import StubModel


def train(n_stages=3):
    torch.manual_seed(0)
    model = StubModel(n_modes_max=4)
    trainer = SimultaneousTrainer(
        model,
        stage_criterion=FixedIterations(30),
        enrichment_criterion=MaxStages(n_stages),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.05),
    )
    history = trainer.enrich()
    return model, history


def test_a_reloaded_model_reproduces_the_trained_one_exactly(tmp_path):
    model, history = train()
    trained = model(None).detach().clone()
    save_checkpoint(tmp_path / "run.pt", model, history)

    fresh = StubModel(n_modes_max=4)
    load_checkpoint(tmp_path / "run.pt", fresh)

    # Bitwise, not approximately: it is the same tensor written and read back.
    assert torch.equal(fresh(None).detach(), trained)


def test_the_truncation_is_restored_with_the_values(tmp_path):
    # The load-bearing detail: a fresh model has one active mode, and nothing in
    # the checkpoint API says how many the run ended with -- the `active` flags
    # ride along inside the state_dict. Without that, a reloaded model would
    # silently evaluate mode 0 only.
    model, history = train(n_stages=3)
    save_checkpoint(tmp_path / "run.pt", model, history)

    fresh = StubModel(n_modes_max=4)
    assert fresh.decomposition.n_modes_truncated == 1
    load_checkpoint(tmp_path / "run.pt", fresh)

    assert fresh.decomposition.n_modes_truncated == 3


def test_the_history_round_trips(tmp_path):
    model, history = train()
    save_checkpoint(tmp_path / "run.pt", model, history, metadata={"strategy": "sim"})

    loaded_history, metadata = load_checkpoint(tmp_path / "run.pt", StubModel())

    assert metadata == {"strategy": "sim"}
    assert [r.stage for r in loaded_history.stages] == [r.stage for r in history.stages]
    assert loaded_history.stages[-1].energy == history.stages[-1].energy
    assert loaded_history.stages[-1].diagnostics == history.stages[-1].diagnostics


def test_training_can_be_resumed_from_a_checkpoint(tmp_path):
    # Trained-then-saved-then-resumed must reach the state of an uninterrupted
    # run. requires_grad is not in a state_dict, so this is really the assertion
    # that prepare_stage sets the whole freeze state and does not inherit it.
    model, saved_history = train(n_stages=2)
    save_checkpoint(tmp_path / "run.pt", model, saved_history)

    resumed = StubModel(n_modes_max=4)
    history, _ = load_checkpoint(tmp_path / "run.pt", resumed)
    trainer = SimultaneousTrainer(
        resumed,
        stage_criterion=FixedIterations(30),
        enrichment_criterion=MaxStages(3),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.05),
    )
    # The reloaded history is what makes enrich() resumable: the next stage
    # index is len(history.stages), so this runs stage 2 and only stage 2.
    trainer.history = history
    trainer.enrich()

    uninterrupted, _ = train(n_stages=3)
    assert len(trainer.history.stages) == 3
    assert resumed(None).detach() == pytest.approx(
        uninterrupted(None).detach(), abs=1e-5
    )


def test_a_mismatched_model_refuses_to_load(tmp_path):
    # Silently loading half a decomposition would produce plausible, wrong plots.
    model, _ = train()
    save_checkpoint(tmp_path / "run.pt", model)

    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / "run.pt", StubModel(n_modes_max=2))


def test_a_foreign_format_is_rejected_with_advice(tmp_path):
    torch.save({"format": 99, "state_dict": {}}, tmp_path / "old.pt")

    with pytest.raises(ValueError, match="Retrain"):
        load_checkpoint(tmp_path / "old.pt", StubModel())


def test_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt", StubModel())
