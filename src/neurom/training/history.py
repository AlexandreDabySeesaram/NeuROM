"""Records of a training run.

Plain data: no torch, no model. The trainer fills these in; examples and tests
read them.
"""

from dataclasses import dataclass, field


@dataclass
class StageRecord:
    """Outcome of one training stage.

    A *stage* is one unit of the training schedule. For the greedy strategy it
    is one mode, but the base trainer does not assume it.

    Attributes:
        stage (int): Index of the stage within the whole run.
        losses (list[float]): Loss at every iteration of the stage. Each value
            is the loss *before* that iteration's update, as returned by
            ``optimizer.step(closure)``.
        phases (list[str]): Name of the phase that ran each iteration, aligned
            with ``losses``.
        stop_reason (str): Why the last phase of the stage ended
            (``"converged"``, ``"max_iter"``, ``"n_iter"``, ``"diverged"``).
        final_energy (float): Loss of the state the stage ended in, evaluated
            once after the last update. None until the stage has run.
    """

    stage: int
    losses: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    stop_reason: str = ""
    final_energy: float = None

    @property
    def n_iter(self) -> int:
        """Number of iterations run in this stage."""
        return len(self.losses)

    @property
    def energy(self) -> float:
        """Energy of the state this stage produced.

        ``final_energy`` when set; ``losses[-1]`` is the loss *before* the last
        update, so it is only a fallback (NaN if there is no loss either).
        """
        if self.final_energy is not None:
            return self.final_energy
        return self.losses[-1] if self.losses else float("nan")


@dataclass
class TrainingHistory:
    """Every stage of a run, in order.

    Accumulates across repeated ``enrich()`` calls, so a run can be resumed.

    Attributes:
        stages (list[StageRecord]): One record per completed stage.
        stop_reason (str): Why the whole run ended (``"n_stages"``,
            ``"diverged"``, ...).
    """

    stages: list = field(default_factory=list)
    stop_reason: str = ""

    def append(self, record: StageRecord) -> None:
        """Append a completed stage."""
        self.stages.append(record)

    @property
    def losses(self) -> list:
        """Every iteration's loss, all stages concatenated (for plotting)."""
        return [value for stage in self.stages for value in stage.losses]
