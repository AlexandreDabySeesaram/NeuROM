"""Unit tests for the trainer loop, against a stub model.

The stub mimics only what the trainer actually depends on: a callable model
whose ``forward`` takes no arguments in training mode, a ``loss(output)``
method, and parameters that can be frozen individually. Deliberately no FEM --
whether the loop is correct is independent of whether the energy is.
"""

import math

import pytest
import torch
import torch.nn as nn

from neurom.training.base import PGDTrainer
from neurom.training.criteria import FixedIterations, MaxStages, RelativeGain


class StubModel(nn.Module):
    """Least-squares stand-in for NeuROMModel.

    Approximates a fixed ``target`` by a sum of scaled directions, one per
    group. Mirrors the structure the trainer cares about:

    * groups are activated one at a time (like PGD modes),
    * an *active but frozen* group still contributes to the loss, exactly as a
      frozen PGD mode still contributes to the energy,
    * a freshly activated group starts at zero, so it contributes nothing until
      trained -- which makes the loss continuous across a stage seam, as the PGD
      energy is.
    """

    def __init__(self):
        super().__init__()
        # Groups 0 and 1 already span the target, so group 2 gains nothing --
        # which is what lets RelativeGain fire before capacity. A 4th group
        # exists so "stopped early on gain" and "ran out of capacity" are
        # distinguishable outcomes.
        self.directions = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, -1.0]),
        ]
        self.groups = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.directions]
        )
        self.register_buffer("target", torch.tensor([1.0, -2.0]))
        self.active = [False] * len(self.groups)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, coords=None):
        return self

    def loss(self, output):
        approximation = torch.zeros(2)
        for group, direction, is_active in zip(
            self.groups, self.directions, self.active
        ):
            if is_active:
                approximation = approximation + group * direction
        return ((approximation - self.target) ** 2).sum()


class StubTrainer(PGDTrainer):
    """Greedy-shaped concrete trainer over StubModel's groups."""

    def prepare_stage(self, stage_index):
        for group in self.model.groups:
            group.requires_grad_(False)
        self.model.active[stage_index] = True
        self.model.groups[stage_index].requires_grad_(True)
        self.make_optimizer()

    def should_add_stage(self, stage_index):
        if stage_index >= len(self.model.groups):
            self.history.stop_reason = "capacity"
            return False
        reason = self.enrichment_criterion.stop_reason(self.history.stages)
        if reason:
            self.history.stop_reason = reason
            return False
        return True


def make_trainer(**kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("stage_criterion", FixedIterations(40))
    kwargs.setdefault("enrichment_criterion", MaxStages(4))
    return StubTrainer(StubModel(), **kwargs)


def test_enrich_runs_one_stage_per_group():
    history = make_trainer().enrich()

    assert len(history.stages) == 4
    assert [record.stage for record in history.stages] == [0, 1, 2, 3]


def test_loss_decreases_within_every_stage():
    history = make_trainer().enrich()

    for record in history.stages:
        assert record.losses[-1] <= record.losses[0]


def test_a_new_stage_starts_from_the_state_the_previous_one_produced():
    # Enriching the approximation space must not move the current solution: a
    # freshly activated group contributes nothing (it starts at zero), so the
    # loss at the seam is exactly the previous stage's final energy. Exact
    # equality, not a tolerance -- it is the same model state evaluated twice.
    history = make_trainer().enrich()

    for previous, current in zip(history.stages, history.stages[1:]):
        assert current.losses[0] == pytest.approx(previous.energy, rel=1e-12)


def test_frozen_groups_do_not_move_in_later_stages():
    # THE defining property of purely greedy training. Every other assertion
    # here would still pass if a later stage quietly retrained an earlier group.
    trainer = make_trainer(enrichment_criterion=MaxStages(1))
    trainer.enrich()
    frozen = trainer.model.groups[0].detach().clone()

    trainer.enrichment_criterion = MaxStages(4)
    trainer.enrich()

    assert torch.equal(trainer.model.groups[0].detach(), frozen)


def test_history_bookkeeping_matches_the_iterations_actually_run():
    history = make_trainer(stage_criterion=FixedIterations(7)).enrich()

    assert all(record.n_iter == 7 for record in history.stages)
    assert len(history.losses) == 7 * len(history.stages)
    assert all(record.stop_reason == "n_iter" for record in history.stages)


def test_gain_is_recorded_for_every_stage_after_the_first():
    history = make_trainer().enrich()

    assert math.isnan(history.stages[0].gain)
    for record in history.stages[1:]:
        assert math.isfinite(record.gain)


def test_enrichment_criterion_can_end_the_run_early():
    # Groups 0 and 1 span the target, so stage 2 gains nothing. RelativeGain
    # must notice and stop before spending group 3 -- i.e. stop for "converged"
    # rather than running to capacity.
    trainer = make_trainer(
        stage_criterion=FixedIterations(300),
        enrichment_criterion=RelativeGain(tol=1e-3),
    )
    history = trainer.enrich()

    assert len(history.stages) == 3
    assert history.stop_reason == "converged"


def test_capacity_ends_the_run_when_no_criterion_fires():
    history = make_trainer(enrichment_criterion=MaxStages(99)).enrich()

    assert len(history.stages) == 4
    assert history.stop_reason == "capacity"


def test_a_non_finite_loss_marks_the_stage_diverged_and_stops_the_run():
    trainer = make_trainer(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e30),
        stage_criterion=FixedIterations(200),
    )
    history = trainer.enrich()

    assert history.stages[-1].diverged
    assert history.stages[-1].stop_reason == "diverged"
    assert history.stop_reason == "diverged"
    # The run stopped instead of building later stages on a poisoned residual.
    assert len(history.stages) < 4


def test_divergence_caused_by_the_final_update_is_still_caught():
    # step() reports the loss BEFORE its update, so a stage whose last update
    # blows the model up would otherwise return a clean record carrying a stale
    # finite energy, and the run would keep building on a poisoned state.
    trainer = make_trainer(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e30),
        stage_criterion=FixedIterations(1),
        enrichment_criterion=MaxStages(1),
    )
    history = trainer.enrich()

    record = history.stages[0]
    assert math.isfinite(record.losses[0])  # the only recorded loss was finite
    assert record.diverged
    assert record.stop_reason == "diverged"
    assert history.stop_reason == "diverged"


def test_enrich_is_resumable():
    # Two calls of one stage each must reach the same state as one call of two.
    incremental = make_trainer(enrichment_criterion=MaxStages(1))
    incremental.enrich()
    incremental.enrichment_criterion = MaxStages(2)
    incremental.enrich()

    single = make_trainer(enrichment_criterion=MaxStages(2))
    single.enrich()

    assert len(incremental.history.stages) == 2
    for resumed, direct in zip(incremental.model.groups, single.model.groups):
        assert resumed.detach() == pytest.approx(direct.detach(), rel=1e-6)


def test_step_returns_the_loss_before_the_update():
    trainer = make_trainer()
    trainer.prepare_stage(0)
    before = trainer.model.loss(trainer.model()).item()

    assert trainer.step() == pytest.approx(before, rel=1e-9)


def test_make_optimizer_refuses_an_empty_parameter_set():
    trainer = make_trainer()
    for group in trainer.model.groups:
        group.requires_grad_(False)

    with pytest.raises(RuntimeError, match="No trainable parameters"):
        trainer.make_optimizer()


def test_each_stage_gets_a_fresh_optimizer():
    trainer = make_trainer()
    trainer.prepare_stage(0)
    first = trainer.optimizer
    trainer.prepare_stage(1)

    assert trainer.optimizer is not first
