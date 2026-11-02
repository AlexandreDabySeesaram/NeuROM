"""Base class for PGD training strategies."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch

from neurom.training.criteria import MaxStages, StageCriterion, Stagnation
from neurom.training.history import StageRecord, TrainingHistory
from neurom.training.progress import ProgressReporter


@dataclass
class Phase:
    """One optimizer run inside a stage, with its own stopping criterion.

    A stage runs its phases in order, e.g. Adam until a plateau, then a few
    LBFGS iterations. Each phase builds a fresh optimizer over the parameters
    trainable at that moment, unless the trainer keeps it (see
    :class:`~neurom.training.simultaneous.SimultaneousTrainer`).

    Attributes:
        optimizer_factory (Callable): ``params -> Optimizer``. LBFGS works too
            (the step uses a closure).
        criterion (StageCriterion): When this phase is done. Checked before
            every iteration on this phase's own losses, so a criterion that is
            already met (e.g. ``FixedIterations(0)``) skips the phase.
        name (str): Label recorded for each iteration in ``StageRecord.phases``.
    """

    optimizer_factory: Callable
    criterion: StageCriterion
    name: str = ""


def _default_phases():
    return [
        Phase(lambda params: torch.optim.Adam(params, lr=0.1), Stagnation(), "adam")
    ]


class PGDTrainer(ABC):
    """Template for a PGD training strategy.

    A run is a sequence of *stages*; each stage runs its :class:`Phase` in
    order, and the run stops when :meth:`should_add_stage` says so.

    The base knows nothing about modes. Everything that touches the
    decomposition happens in :meth:`prepare_stage`, and the trainable set is
    read off the freeze state (``requires_grad``), so the same loop can drive
    strategies whose stages are not modes.

    Args:
        model (NeuROMModel): Model to train, callable with no argument in
            training mode and exposing ``loss(output)``.
        phases (list[Phase], optional): The phases of every stage. Defaults to
            a single Adam(lr=0.1) phase stopped by
            :class:`~neurom.training.criteria.Stagnation` (defaults of Algorithm 1).
        enrichment_criterion (EnrichmentCriterion, optional): When to stop
            starting stages. Defaults to ``MaxStages(1)``.
        progress (ProgressReporter, optional): Told about every stage, phase
            and iteration, e.g. :class:`~neurom.training.progress.ProgressLog`.
            Defaults to the silent ``ProgressReporter()``.
        final_phases (list[Phase], optional): Run once after the enrichment,
            e.g. the LBFGS polish of §3.2.2 in Daby-Seesaram et al. (2026),
            Part II. They train what the last stage left trainable (every mode
            for the simultaneous strategy, the last mode only for the greedy
            one) and add no mode. Defaults to none.

    Attributes:
        optimizer (torch.optim.Optimizer): The current phase's optimizer, None
            before the first phase.
        history (TrainingHistory): Accumulates across :meth:`enrich` calls.
    """

    def __init__(
        self, model, phases=None, enrichment_criterion=None, progress=None, final_phases=None
    ):
        self.model = model
        self.phases = phases or _default_phases()
        self.enrichment_criterion = enrichment_criterion or MaxStages(1)
        self.progress = progress or ProgressReporter()
        self.final_phases = final_phases or []
        self.optimizer = None
        self.history = TrainingHistory()

    def trainable_parameters(self):
        """The parameters the current freeze state leaves trainable.

        Returns:
            list[torch.nn.Parameter]: Every parameter of the model with
            ``requires_grad`` set.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def make_optimizer(self, optimizer_factory):
        """Build a fresh optimizer over the currently trainable parameters.

        A fresh optimizer per phase means no optimizer state (e.g. Adam's
        moments) is carried from one phase or stage to the next.

        Args:
            optimizer_factory (Callable): ``params -> Optimizer``.

        Returns:
            torch.optim.Optimizer: The new optimizer, also stored on
            ``self.optimizer``.

        Raises:
            RuntimeError: If nothing is trainable.
        """
        params = self.trainable_parameters()
        if not params:
            raise RuntimeError(
                "No trainable parameters: prepare_stage left everything frozen."
            )
        self.optimizer = optimizer_factory(params)
        return self.optimizer

    def enrich(self):
        """Run stages until :meth:`should_add_stage` stops or a stage diverges,
        then the final phases (see :meth:`finish`).

        Resumable: the next stage index is ``len(self.history.stages)``. The
        final phases run again at the end of every call.

        Returns:
            TrainingHistory: The accumulated history (also ``self.history``).
        """
        self.model.train()
        while self.should_add_stage(len(self.history.stages)):
            stage_index = len(self.history.stages)
            self.prepare_stage(stage_index)
            record = self.stage(stage_index)
            self.history.append(record)
            self.progress.stage_end(record)
            if record.stop_reason == "diverged":
                self.history.stop_reason = "diverged"
                break
        if self.history.stop_reason != "diverged":
            self.finish()
        self.progress.close(self.history)
        return self.history

    def stage(self, stage_index):
        """Run every phase in order, then evaluate the final energy.

        A non-finite loss ends the stage with ``"diverged"``, skipping the
        remaining phases.

        Args:
            stage_index (int): Index of this stage within the run.

        Returns:
            StageRecord: The stage's losses and outcome.
        """
        record = StageRecord(stage=stage_index)
        self.progress.stage_start(stage_index)
        return self._run_phases(record, self.phases)

    def finish(self):
        """Run the final phases once, without adding a stage.

        Their record goes to ``history.final``, not to ``history.stages``, so
        the stage count still equals the mode count.

        Returns:
            StageRecord or None: The final record, None without final phases.
        """
        if not self.final_phases:
            return None
        record = StageRecord(stage=len(self.history.stages))
        self.progress.final_start()
        self._run_phases(record, self.final_phases)
        self.history.final = record
        self.progress.final_end(record)
        if record.stop_reason == "diverged":
            self.history.stop_reason = "diverged"
        return record

    def _run_phases(self, record, phases):
        """Run ``phases`` in order into ``record``, then evaluate its final energy.

        Args:
            record (StageRecord): Filled in place.
            phases (list[Phase]): The phases to run.

        Returns:
            StageRecord: ``record``.
        """
        for phase in phases:
            self.make_optimizer(phase.optimizer_factory)
            self.progress.phase_start(phase.name, getattr(phase.criterion, "max_iter", None))
            losses = []
            while not (reason := phase.criterion.stop_reason(losses)):
                loss = self.step()
                losses.append(loss)
                record.losses.append(loss)
                record.phases.append(phase.name)
                self.progress.update(len(losses), loss)
                if not math.isfinite(loss):
                    record.stop_reason = "diverged"
                    record.final_energy = loss
                    self.progress.phase_end(phase.name, len(losses), "diverged", loss)
                    return record
            record.stop_reason = reason
            self.progress.phase_end(phase.name, len(losses), reason, losses[-1] if losses else None)
        record.final_energy = float(self.model.loss(self.model()).detach())
        if not math.isfinite(record.final_energy):
            record.stop_reason = "diverged"
        return record

    def step(self):
        """Run one optimizer iteration.

        Returns:
            float: The loss **before** this iteration's update, which is what
            ``optimizer.step(closure)`` returns.
        """
        return float(self.optimizer.step(self._closure()).detach())

    def _closure(self):
        """Build the closure ``optimizer.step`` calls.

        Returns:
            Callable[[], torch.Tensor]: Zeroes the gradients, runs a forward,
            evaluates the loss, backpropagates and returns the loss.
        """

        def closure():
            self.optimizer.zero_grad()
            loss = self.model.loss(self.model())
            loss.backward()
            return loss

        return closure

    @abstractmethod
    def prepare_stage(self, stage_index):
        """Set the freeze state for this stage.

        The only place the decomposition is manipulated. The optimizers are
        built afterwards, one per phase, on what this leaves trainable.

        Args:
            stage_index (int): Index of the stage about to run.
        """

    def should_add_stage(self, stage_index):
        """Run stage 0 unconditionally, then ask the enrichment criterion.

        Overrides that stop should set ``self.history.stop_reason``.

        Args:
            stage_index (int): Index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """
        if stage_index == 0:
            return True
        reason = self.enrichment_criterion.stop_reason(self.history.stages)
        if reason:
            self.history.stop_reason = reason
            return False
        return True
