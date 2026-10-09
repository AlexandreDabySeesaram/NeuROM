"""Base class for PGD training strategies."""

import math
from abc import ABC, abstractmethod

import torch

from neurom.training.criteria import MaxStages, RelativeChange
from neurom.training.history import StageRecord, TrainingHistory


def _default_optimizer(params):
    return torch.optim.Adam(params, lr=0.1)


class PGDTrainer(ABC):
    """Template for a PGD training strategy.

    A run is a sequence of *stages*; each stage iterates :meth:`step` until the
    stage criterion fires, and the run stops when :meth:`should_add_stage`
    says so.

    The base knows nothing about modes. Everything that touches the
    decomposition happens in :meth:`prepare_stage`, and the trainable set is
    read off the freeze state (``requires_grad``), so the same loop can drive
    strategies whose stages are not modes.

    Args:
        model (NeuROMModel): Model to train, callable with no argument in
            training mode and exposing ``loss(output)``.
        optimizer_factory (Callable, optional): ``params -> Optimizer``, called
            once per stage on the currently trainable parameters. Defaults to
            ``Adam(lr=0.1)``. LBFGS works too (the step uses a closure).
        stage_criterion (StageCriterion, optional): When a stage is done.
            Defaults to :class:`~neurom.training.criteria.RelativeChange`.
        enrichment_criterion (EnrichmentCriterion, optional): When to stop
            starting stages. Defaults to ``MaxStages(1)``.

    Attributes:
        optimizer (torch.optim.Optimizer): The current stage's optimizer, None
            before the first stage.
        history (TrainingHistory): Accumulates across :meth:`enrich` calls.
    """

    def __init__(
        self,
        model,
        optimizer_factory=None,
        stage_criterion=None,
        enrichment_criterion=None,
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory or _default_optimizer
        self.stage_criterion = stage_criterion or RelativeChange()
        self.enrichment_criterion = enrichment_criterion or MaxStages(1)
        self.optimizer = None
        self.history = TrainingHistory()

    def trainable_parameters(self):
        """The parameters the current freeze state leaves trainable.

        Returns:
            list[torch.nn.Parameter]: Every parameter of the model with
            ``requires_grad`` set.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def make_optimizer(self):
        """Build a fresh optimizer over the currently trainable parameters.

        A fresh optimizer per stage means no optimizer state (e.g. Adam's
        moments) is carried from one stage to the next.

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
        self.optimizer = self.optimizer_factory(params)
        return self.optimizer

    def enrich(self):
        """Run stages until :meth:`should_add_stage` stops or a stage diverges.

        Resumable: the next stage index is ``len(self.history.stages)``.

        Returns:
            TrainingHistory: The accumulated history (also ``self.history``).
        """
        self.model.train()
        while self.should_add_stage(len(self.history.stages)):
            stage_index = len(self.history.stages)
            self.prepare_stage(stage_index)
            record = self.stage(stage_index)
            self.history.append(record)
            if record.stop_reason == "diverged":
                self.history.stop_reason = "diverged"
                break
        return self.history

    def stage(self, stage_index):
        """Iterate :meth:`step` until the stage criterion fires.

        A non-finite loss ends the stage with ``"diverged"``.

        Args:
            stage_index (int): Index of this stage within the run.

        Returns:
            StageRecord: The stage's losses and outcome.
        """
        record = StageRecord(stage=stage_index)
        while True:
            loss = self.step()
            record.losses.append(loss)
            if not math.isfinite(loss):
                record.stop_reason = "diverged"
                record.final_energy = loss
                return record
            reason = self.stage_criterion.stop_reason(record.losses)
            if reason:
                record.stop_reason = reason
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
        """Set the freeze state for this stage and build its optimizer.

        The only place the decomposition is manipulated. Must leave an
        optimizer in place (usually by calling :meth:`make_optimizer`).

        Args:
            stage_index (int): Index of the stage about to run.
        """

    @abstractmethod
    def should_add_stage(self, stage_index):
        """Whether to run stage ``stage_index``.

        Implementations that stop should set ``self.history.stop_reason``.

        Args:
            stage_index (int): Index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """
