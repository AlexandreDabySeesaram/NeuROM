"""Stopping criteria for PGD training.

Two protocols, because the two decisions see different data: a stage criterion
watches the per-iteration losses of one stage, an enrichment criterion watches
the completed stages. They are injected into the trainer rather than overridden
as methods, so the stopping rule and the training strategy stay independent.

Each returns a short reason string to stop, or None to continue, so the history
can say *why* a stage or a run ended.
"""

from abc import ABC, abstractmethod


class StageCriterion(ABC):
    """Decides when one training stage has finished."""

    @abstractmethod
    def stop_reason(self, losses):
        """Return a short reason to stop, or None to continue.

        Args:
            losses (list[float]): Every loss recorded so far in this stage.
        """


class EnrichmentCriterion(ABC):
    """Decides when to stop starting new stages."""

    @abstractmethod
    def stop_reason(self, stages):
        """Return a short reason to stop, or None to continue.

        Args:
            stages (list[StageRecord]): Completed stages, in order.
        """


class RelativeChange(StageCriterion):
    """Stop when the relative change of the loss between two iterations is small.

    Stops with ``"converged"`` when
    ``|L_k - L_{k-1}| <= tol * max(|L_k|, 1e-30)``, or with ``"max_iter"`` after
    ``max_iter`` iterations.

    Args:
        tol (float): Relative-change threshold.
        max_iter (int): Hard cap, so a stage always terminates.
    """

    def __init__(self, tol=1e-6, max_iter=1000):
        self.tol = tol
        self.max_iter = max_iter

    def stop_reason(self, losses):
        if len(losses) >= self.max_iter:
            return "max_iter"
        if len(losses) < 2:
            return None
        current, previous = losses[-1], losses[-2]
        if abs(current - previous) <= self.tol * max(abs(current), 1e-30):
            return "converged"
        return None


class FixedIterations(StageCriterion):
    """Run exactly ``n_iter`` iterations per stage.

    Reproducible, blind to convergence: reproduces a hand-written loop of
    ``n_iter`` optimizer steps.

    Args:
        n_iter (int): Iterations to run.
    """

    def __init__(self, n_iter):
        self.n_iter = n_iter

    def stop_reason(self, losses):
        return "n_iter" if len(losses) >= self.n_iter else None


class MaxStages(EnrichmentCriterion):
    """Stop after ``n_stages`` stages.

    Args:
        n_stages (int): Stage budget.
    """

    def __init__(self, n_stages):
        self.n_stages = n_stages

    def stop_reason(self, stages):
        return "n_stages" if len(stages) >= self.n_stages else None
