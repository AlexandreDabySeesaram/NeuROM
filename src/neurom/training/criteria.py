"""Stopping criteria for PGD training.

Two protocols, because the two decisions see different data: a stage criterion
watches a stream of per-iteration losses, an enrichment criterion watches
completed stages. Injected into the trainer rather than overridden as methods,
so the stopping rule and the training strategy stay independent -- otherwise
every combination needs its own subclass (GreedyWithRelTol, GreedyWithFixedIter,
AlternatingWithRelTol, ...).

Each returns a short reason string to stop, or None to continue, so the history
can say *why* a run ended.
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

    def budget(self):
        """Iterations this criterion can run at most, or None if unbounded.

        Advisory only -- nothing in the loop reads it to decide anything; it is
        what lets a :class:`~neurom.training.progress.ProgressBar` show a
        fraction instead of a bare counter. A criterion with no hard cap says
        None rather than guessing, and the bar degrades to a counter.
        """
        return None


class EnrichmentCriterion(ABC):
    """Decides when to stop starting new stages."""

    @abstractmethod
    def stop_reason(self, stages):
        """Return a short reason to stop, or None to continue.

        Args:
            stages (list[StageRecord]): Completed stages, in order.
        """


class RelativeChange(StageCriterion):
    """Stop when the loss plateaus over a sliding window.

    Improvement is measured as ``(past - current) / denom`` over ``window``
    iterations, with ``denom = max(|current|, |past|, floor)``.

    Three details are load-bearing, all learned from the beam problem (see
    ``tests/integration/test_1d_beam_deflection_PGD.py``):

    * ``min_iter`` -- Adam has a long sticky early phase (~100 iterations) where
      the energy barely moves before it escapes and dives. Without a floor on
      the iteration count a plateau detector mistakes that for convergence.
    * ``floor`` -- the energy crosses zero (~+2e5 to ~-1e8) and spans some eight
      orders of magnitude, so a purely relative denominator blows up near the
      crossing. Flooring it at 1.0 keeps the ratio meaningful.
    * the improvement is **signed**, so a rising loss reads as no progress and
      stops the stage rather than looking like a large change.

    Args:
        tol (float): Relative-improvement threshold below which we stop.
        window (int): Number of iterations over which improvement is measured.
        max_iter (int): Hard cap so a stage always terminates.
        min_iter (int): Iterations that must run before the plateau test applies.
        floor (float): Lower bound on the relative denominator.
    """

    def __init__(self, tol=1e-4, window=20, max_iter=1000, min_iter=100, floor=1.0):
        self.tol = tol
        self.window = window
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.floor = floor

    def stop_reason(self, losses):
        n = len(losses)
        if n >= self.max_iter:
            return "max_iter"
        if n < self.min_iter or n <= self.window:
            return None
        current = losses[-1]
        past = losses[-1 - self.window]
        denom = max(abs(current), abs(past), self.floor)
        if (past - current) / denom < self.tol:
            return "converged"
        return None

    def budget(self):
        """``max_iter`` -- the hard cap. Most stages converge well before it."""
        return self.max_iter


class FixedIterations(StageCriterion):
    """Run exactly ``n_iter`` iterations per stage.

    Reproduces the fixed-length training loop the examples used before the
    trainer existed. Fully reproducible; blind to whether anything converged.

    Args:
        n_iter (int): Iterations to run.
    """

    def __init__(self, n_iter):
        self.n_iter = n_iter

    def stop_reason(self, losses):
        return "n_iter" if len(losses) >= self.n_iter else None

    def budget(self):
        """``n_iter`` -- exact, not a cap."""
        return self.n_iter


class RelativeGain(EnrichmentCriterion):
    """Stop enriching when a stage barely improved on the previous one.

    Args:
        tol (float): Relative-gain threshold below which enrichment stops.
        floor (float): Lower bound on the relative denominator, for the same
            zero-crossing reason as :class:`RelativeChange`.
    """

    def __init__(self, tol=1e-4, floor=1.0):
        self.tol = tol
        self.floor = floor

    def stop_reason(self, stages):
        if len(stages) < 2:
            return None
        previous = stages[-2].energy
        current = stages[-1].energy
        denom = max(abs(current), abs(previous), self.floor)
        if (previous - current) / denom < self.tol:
            return "converged"
        return None


class MaxStages(EnrichmentCriterion):
    """Stop after ``n_stages`` stages, whatever the gains.

    Args:
        n_stages (int): Stage budget.
    """

    def __init__(self, n_stages):
        self.n_stages = n_stages

    def stop_reason(self, stages):
        return "n_stages" if len(stages) >= self.n_stages else None
