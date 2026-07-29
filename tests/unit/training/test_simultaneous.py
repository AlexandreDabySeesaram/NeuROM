"""Unit tests for the simultaneous strategy, against a stub CP decomposition.

The stub is a real rank-``m`` CP least-squares problem -- a sum of outer
products fitted to a fixed order-3 tensor -- but with no FEM behind it: the
question here is whether the *schedule* is right (what is trainable when, what
moves, what the diagnostics see), not whether an energy is.

The same stub drives ``GreedyTrainer``, so the two strategies are compared on
identical numbers rather than described side by side.
"""

import pytest
import torch
import torch.nn as nn

from neurom.training import FixedIterations, GreedyTrainer, MaxStages
from neurom.training.simultaneous import SimultaneousTrainer

SHAPE = (4, 3, 2)


class StubField:
    """The slice of ``TrainableField`` the trainers and diagnostics touch."""

    def __init__(self, size):
        self.values_reduced = nn.Parameter(torch.empty(size))

    def full_values(self):
        return self.values_reduced


class StubCP(nn.Module):
    """CP decomposition of an order-3 tensor, with the CPPGD lifecycle.

    Reproduces exactly the API the trainers use: ``monoms``, ``add_mode``,
    ``freeze_mode``/``unfreeze_mode``/``freeze_all``, ``renormalise``,
    ``n_modes_truncated`` and ``n_modes_max``. A mode is *active* (contributes to the reconstruction) and
    separately *frozen* (contributes but does not move), the same two
    independent flags as the real decomposition.
    """

    def __init__(self, n_modes_max=4, shape=SHAPE):
        super().__init__()
        self.n_modes_max = n_modes_max
        self.shape = shape
        self.monoms = [[StubField(size) for size in shape] for _ in range(n_modes_max)]
        # Registered so model.parameters() -- what the trainer reads the freeze
        # state off -- sees every monom, active or not.
        self._monom_parameters = nn.ParameterList(
            [field.values_reduced for mode in self.monoms for field in mode]
        )
        # A registered buffer, exactly as QuadratureAssembly.active is: which
        # modes are active must ride along in the state_dict, or a reloaded
        # checkpoint would silently evaluate mode 0 only.
        self.register_buffer("active", torch.zeros(n_modes_max, dtype=torch.bool))
        self.reset()

    def reset(self):
        """Seed every mode identically, as ``Axis.init_values`` does."""
        with torch.no_grad():
            for mode in self.monoms:
                for field in mode:
                    field.values_reduced.fill_(0.5)
            self.active.zero_()
            self.active[0] = True
        self.freeze_all()
        self.unfreeze_mode(0)

    @property
    def n_modes_truncated(self):
        n = 0
        for is_active in self.active:
            if not is_active:
                break
            n += 1
        return n

    def freeze_all(self):
        for m in range(self.n_modes_max):
            self.freeze_mode(m)

    def freeze_mode(self, m):
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(False)

    def unfreeze_mode(self, m):
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(True)

    def renormalise(self):
        """Gauge-fixing hook the trainers call at stage boundaries.

        A no-op here, as it is on ``CPPGD``: this stub has the same scale
        degeneracy, but the trainer tests are about the mode lifecycle, not the
        gauge.
        """

    def add_mode(self):
        m = self.n_modes_truncated
        if m >= self.n_modes_max:
            raise RuntimeError("Cannot add a mode: all modes are already active.")
        with torch.no_grad():
            self.active[m] = True
        self.unfreeze_mode(m)
        return m

    def reconstruct(self):
        total = torch.zeros(self.shape)
        for m in range(self.n_modes_truncated):
            a, b, c = (field.values_reduced for field in self.monoms[m])
            total = total + a[:, None, None] * b[None, :, None] * c[None, None, :]
        return total


class StubModel(nn.Module):
    """Least-squares fit of a fixed tensor by the CP decomposition."""

    def __init__(self, n_modes_max=4):
        super().__init__()
        self.decomposition = StubCP(n_modes_max=n_modes_max)
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("target", torch.randn(SHAPE, generator=generator))

    def forward(self, coords=None):
        return self.decomposition.reconstruct()

    def loss(self, output):
        return ((output - self.target) ** 2).sum()


def make_trainer(cls=SimultaneousTrainer, n_modes_max=4, n_iter=200, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("stage_criterion", FixedIterations(n_iter))
    kwargs.setdefault("enrichment_criterion", MaxStages(n_modes_max))
    kwargs.setdefault("optimizer_factory", lambda p: torch.optim.Adam(p, lr=0.05))
    return cls(StubModel(n_modes_max=n_modes_max), **kwargs)


def mode_values(trainer, mode):
    return [
        field.values_reduced.detach().clone()
        for field in trainer.model.decomposition.monoms[mode]
    ]


def test_every_active_mode_is_trainable_during_a_stage():
    trainer = make_trainer()
    trainer.prepare_stage(0)
    trainer.prepare_stage(1)
    trainer.prepare_stage(2)
    decomposition = trainer.model.decomposition

    assert decomposition.n_modes_truncated == 3
    for m in range(3):
        for field in decomposition.monoms[m]:
            assert field.values_reduced.requires_grad
    # ... and nothing beyond the active block was woken up.
    for field in decomposition.monoms[3]:
        assert not field.values_reduced.requires_grad


def test_a_stage_that_arrives_frozen_is_unfrozen_anyway():
    # add_mode() only unfreezes the mode it adds, so a decomposition handed over
    # frozen (from a greedy run, say) would otherwise train one mode and call it
    # simultaneous.
    trainer = make_trainer()
    trainer.model.decomposition.freeze_all()
    trainer.prepare_stage(1)

    assert len(trainer.trainable_parameters()) == 2 * len(SHAPE)


def test_earlier_modes_keep_moving_in_later_stages():
    # THE defining property, and the exact mirror of the greedy trainer's
    # test_frozen_groups_do_not_move_in_later_stages.
    trainer = make_trainer(enrichment_criterion=MaxStages(1))
    trainer.enrich()
    after_first_stage = mode_values(trainer, 0)

    trainer.enrichment_criterion = MaxStages(3)
    trainer.enrich()

    assert not any(
        torch.equal(before, after)
        for before, after in zip(after_first_stage, mode_values(trainer, 0))
    )


def test_a_new_stage_starts_from_the_state_the_previous_one_produced():
    # Enrichment must not move the solution discontinuously: the new mode's seed
    # is what it is, but every earlier mode enters the stage exactly as the
    # previous stage left it.
    trainer = make_trainer(enrichment_criterion=MaxStages(1))
    trainer.enrich()
    handed_over = mode_values(trainer, 0)

    trainer.prepare_stage(1)

    for before, at_stage_start in zip(handed_over, mode_values(trainer, 0)):
        assert torch.equal(before, at_stage_start)


def test_it_is_no_worse_than_greedy_at_equal_rank():
    # Greedy's rank-m answer is feasible for the rank-m problem this strategy
    # solves, so it cannot do better -- up to optimiser noise, hence the slack.
    greedy = make_trainer(cls=GreedyTrainer)
    simultaneous = make_trainer(cls=SimultaneousTrainer)

    greedy_energy = greedy.enrich().stages[-1].energy
    simultaneous_energy = simultaneous.enrich().stages[-1].energy

    assert simultaneous_energy <= greedy_energy * (1.0 + 1e-6)


def test_diagnostics_report_the_worst_pair_not_just_the_newest_mode():
    trainer = make_trainer(enrichment_criterion=MaxStages(3))
    history = trainer.enrich()

    for record in history.stages:
        assert record.diagnostics["amplitude"] > 0.0
        assert 0.0 <= record.diagnostics["max_correlation"] <= 1.0 + 1e-6

    # Modes 0 and 1 are made copies of each other and the newest mode is left
    # alone: a newest-mode-only diagnostic would report this state as clean.
    decomposition = trainer.model.decomposition
    with torch.no_grad():
        for target, source in zip(decomposition.monoms[1], decomposition.monoms[0]):
            target.values_reduced.copy_(source.values_reduced)
    record = history.stages[-1]
    trainer.on_stage_end(record)

    assert record.diagnostics["max_correlation"] == pytest.approx(1.0, rel=1e-6)


def test_the_run_stops_at_capacity():
    trainer = make_trainer(n_modes_max=3, n_iter=20, enrichment_criterion=MaxStages(99))
    history = trainer.enrich()

    assert len(history.stages) == 3
    assert history.stop_reason == "capacity"


def test_each_stage_gets_a_fresh_optimizer():
    # Adam's moments for the old modes were accumulated against the energy at a
    # lower truncation order; reusing them would resume those modes with stale
    # second-moment estimates.
    trainer = make_trainer()
    trainer.prepare_stage(0)
    first = trainer.optimizer
    trainer.prepare_stage(1)

    assert trainer.optimizer is not first
