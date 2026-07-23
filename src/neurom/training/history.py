"""Records of a training run.

Plain data: no torch, no model. The trainer fills these in; examples and tests
read them.
"""

from dataclasses import dataclass, field


@dataclass
class StageRecord:
    """Outcome of one training stage.

    A *stage* is one unit of the training schedule. For the greedy strategy that
    is one mode, but the base trainer does not assume it -- a strategy may use a
    stage for something that is not a mode at all (see ``PGDTrainer``).

    Attributes:
        stage (int): Index of the stage within the whole run.
        losses (list[float]): Loss at every iteration of this stage. Each value
            is the loss *before* that iteration's parameter update -- see
            :meth:`neurom.training.base.PGDTrainer.step`.
        stop_reason (str): Why the stage ended (``"converged"``, ``"max_iter"``,
            ``"n_iter"``, ``"diverged"``).
        diverged (bool): True if a non-finite loss ended the stage.
        gain (float): Improvement over the previous stage's final energy. Set by
            :meth:`TrainingHistory.append`; NaN for the first stage.
        diagnostics (dict): Strategy-specific extras. The base trainer never
            writes here; concrete strategies fill it in ``on_stage_end``.
    """

    stage: int
    losses: list = field(default_factory=list)
    stop_reason: str = ""
    diverged: bool = False
    gain: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_iter(self) -> int:
        """Number of iterations run in this stage."""
        return len(self.losses)

    @property
    def energy(self) -> float:
        """Final loss of the stage; NaN if no iteration ran."""
        return self.losses[-1] if self.losses else float("nan")


@dataclass
class TrainingHistory:
    """Every stage of a run, in order.

    Accumulates across repeated ``enrich()`` calls so a run can be resumed.

    Attributes:
        stages (list[StageRecord]): One record per completed stage.
        stop_reason (str): Why the whole run ended (``"capacity"``,
            ``"converged"``, ``"n_stages"``, ``"diverged"``).
    """

    stages: list = field(default_factory=list)
    stop_reason: str = ""

    def append(self, record: StageRecord) -> None:
        """Append ``record``, filling in its gain over the previous stage."""
        if self.stages:
            record.gain = self.stages[-1].energy - record.energy
        self.stages.append(record)

    @property
    def losses(self) -> list:
        """Every iteration's loss, all stages concatenated -- for plotting."""
        return [value for stage in self.stages for value in stage.losses]
