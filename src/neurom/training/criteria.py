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

    def start(self, stage_index):
        """A phase using this criterion is about to run; no-op by default.

        The trainer calls it before every phase, so a criterion can depend on
        the stage (e.g. a looser tolerance for the first modes). The final
        phases get the number of stages run so far.

        Args:
            stage_index (int): Index of the stage the phase belongs to.
        """

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


def _relative_decrease(previous, current):
    """``2 (L_{k-1} - L_k) / |L_{k-1} + L_k|``, the loss decrease of eq. (16) in
    Daby-Seesaram et al. (2026), Part II. Positive when the loss went down."""
    return (previous - current) / max(abs(0.5 * (previous + current)), 1e-30)


class Stagnation(StageCriterion):
    """Stop when the loss stagnates, the mode-addition rule of Algorithm 1 in
    Daby-Seesaram et al. (2026), Part II.

    With ``d_k`` the relative decrease ``2 (L_{k-1} - L_k) / |L_{k-1} + L_k|``,
    an iteration stagnates when ``|d_k| < tol``. Stops with:

    * ``"stagnated"`` after more than ``patience`` stagnating iterations in a
      row;
    * ``"no_improvement"`` after more than ``no_improvement`` iterations in a
      row without a new minimum of the loss (an extra trigger, not in the
      paper);
    * ``"max_iter"`` after ``max_iter`` iterations.

    With ``patience=0`` and ``no_improvement=None``, a stage stops at the first
    stagnating iteration: a plain relative-change test.

    The paper leaves ``tol`` and ``patience`` free, and reports the result
    insensitive to the patience between 5 and 100.

    ``tol`` may depend on the stage ``m`` (the mode the stage adds), e.g. to
    converge the first modes more coarsely::

        Stagnation(tol=lambda m: 1e-3 if m < 3 else 1e-4)

    The function is evaluated by :meth:`start`; before any call, the threshold
    is ``tol(0)``.

    Args:
        tol (float or Callable[[int], float]): Stagnation threshold ``eta_c``
            on ``|d_k|``, or a function of the stage index returning it.
        patience (int): Stagnating iterations tolerated in a row.
        no_improvement (int or None): Iterations without a new minimum
            tolerated in a row; None disables this trigger.
        max_iter (int): Hard cap, so a stage always terminates.
    """

    def __init__(self, tol=1e-4, patience=5, no_improvement=90, max_iter=1000):
        self.tol = tol
        self.stage_tol = tol(0) if callable(tol) else tol
        self.patience = patience
        self.no_improvement = no_improvement
        self.max_iter = max_iter

    def start(self, stage_index):
        """Evaluate the threshold of this stage when ``tol`` is a function.

        Args:
            stage_index (int): Index of the stage about to run.
        """
        self.stage_tol = self.tol(stage_index) if callable(self.tol) else self.tol

    def stop_reason(self, losses):
        if len(losses) >= self.max_iter:
            return "max_iter"
        # Count the trailing stagnating iterations, no further than needed.
        count = 0
        for k in range(len(losses) - 1, 0, -1):
            if abs(_relative_decrease(losses[k - 1], losses[k])) >= self.stage_tol:
                break
            count += 1
            if count > self.patience:
                return "stagnated"
        if self.no_improvement is not None:
            window = self.no_improvement + 1
            if len(losses) > window and min(losses[-window:]) >= min(losses[:-window]):
                return "no_improvement"
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


class RelativeGain(EnrichmentCriterion):
    """Stop enriching when a stage barely improved on the previous one.

    Stops with ``"converged"`` when
    ``(E_{k-1} - E_k) / max(|E_k|, |E_{k-1}|, floor) < tol``, with ``E_k`` the
    energy of stage ``k``. Never stops before two stages have run. It sets no
    stage budget: combine it with :class:`MaxStages` through :class:`AnyOf`.

    Args:
        tol (float): Relative-gain threshold below which enrichment stops.
        floor (float): Lower bound on the relative denominator, so energies
            close to zero do not blow up the relative gain.
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


class Usefulness(EnrichmentCriterion):
    """Stop enriching when the last mode did not help, the stopping rule of
    Algorithm 1 in Daby-Seesaram et al. (2026), Part II.

    A stage is *useful* when its loss went down for ``min_useful`` iterations
    in a row, each a relative decrease ``d_k >= tol`` (see
    :class:`Stagnation`). A stagnating iteration (``|d_k| < tol``) restarts the
    count; an increase neither counts nor restarts it. Stops with
    ``"useless_mode"`` when the last stage was not useful. The first stage is
    never judged, so at least two stages run.

    The paper writes the test as ``L > 0``, read here as a decrease of the
    loss.

    Args:
        tol (float): Stagnation threshold, the ``tol`` of :class:`Stagnation`.
        min_useful (int): Decreasing iterations in a row that make a stage
            useful.
    """

    def __init__(self, tol=1e-4, min_useful=15):
        self.tol = tol
        self.min_useful = min_useful

    def stop_reason(self, stages):
        if len(stages) < 2:
            return None
        run = 0
        losses = stages[-1].losses
        for previous, current in zip(losses, losses[1:]):
            decrease = _relative_decrease(previous, current)
            if abs(decrease) < self.tol:
                run = 0
            elif decrease > 0:
                run += 1
                if run >= self.min_useful:
                    return None
        return "useless_mode"


class MaxStages(EnrichmentCriterion):
    """Stop after ``n_stages`` stages.

    Args:
        n_stages (int): Stage budget.
    """

    def __init__(self, n_stages):
        self.n_stages = n_stages

    def stop_reason(self, stages):
        return "n_stages" if len(stages) >= self.n_stages else None


class AnyOf(EnrichmentCriterion):
    """Stop as soon as any of the given criteria asks to stop.

    The criteria are queried in order; the reason of the first one that stops
    is returned, so the history still says which rule ended the run.

    Args:
        *criteria (EnrichmentCriterion): Criteria to combine, e.g.
            ``AnyOf(RelativeGain(1e-4), MaxStages(10))``.
    """

    def __init__(self, *criteria):
        self.criteria = criteria

    def stop_reason(self, stages):
        for criterion in self.criteria:
            reason = criterion.stop_reason(stages)
            if reason:
                return reason
        return None
