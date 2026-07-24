"""PGD training that keeps every mode trainable."""

from neurom.training import diagnostics
from neurom.training.base import PGDTrainer


class SimultaneousTrainer(PGDTrainer):
    """Add one mode per stage and retrain **all** active modes together.

    One stage is still one enrichment, but nothing is ever frozen: stage ``m``
    minimises the energy over the whole rank-``m+1`` decomposition, using the
    previous stage's modes as the starting point rather than as a fixed
    background.

    The difference from :class:`~neurom.training.greedy.GreedyTrainer` is what
    the new mode is allowed to fix. Greedy freezes its predecessors, so mode
    ``m`` can only descend the residual they left, and any error *inside* an
    earlier mode is permanent -- the classical progressive-Galerkin PGD. Here the
    earlier modes can be revised in the light of the new one, so the stage
    solves the full rank-``m+1`` problem. That is strictly the better minimiser
    at equal rank (greedy's answer is feasible for it, so the optimum is at
    least as good), at the cost of re-optimising every mode at every stage and
    of losing the greedy sequence's one structural guarantee.

    Two consequences worth expecting:

    * **Enrichment gains are not comparable to greedy's.** A stage improves the
      energy both by adding a mode and by correcting the old ones, so
      :class:`~neurom.training.criteria.RelativeGain` sees the sum of the two
      and will keep enriching slightly longer for the same ``tol``.
    * **Modes can collapse onto one another at any time.** Freezing is what
      keeps greedy's modes apart; without it two modes that are free to move
      can converge to the same rank-1 tensor, and not only the newest pair --
      hence the pairwise diagnostic in :meth:`on_stage_end`.

    ``n_modes_ini > 1`` buys nothing here either: every initially-active mode
    starts from the same ``Axis.init_values`` seed and sees the same gradient,
    so stage 0 keeps them parallel. Modes only differentiate once they are
    added at *different* stages, from different states.

    Args:
        model (NeuROMModel): Model whose ``decomposition`` is a ``CPPGD``.
        optimizer_factory, stage_criterion, enrichment_criterion: See
            :class:`~neurom.training.base.PGDTrainer`.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.decomposition = model.decomposition

    def prepare_stage(self, stage_index):
        """Add a mode, unfreeze every active mode, and build a fresh optimizer.

        No ``freeze_all()``: that single omission against
        :meth:`~neurom.training.greedy.GreedyTrainer.prepare_stage` is the whole
        strategy. The explicit unfreeze loop is not redundant with it --
        ``add_mode`` only unfreezes the mode it adds, and the decomposition may
        arrive frozen (from a previous greedy run, or from a strategy that
        alternates), so the stage states its own requirement rather than
        inheriting whatever freeze state it was handed.

        The fresh optimizer per stage matters more here than anywhere else:
        Adam's moment buffers for the old modes were accumulated against the
        energy at a lower truncation order, and reusing them would resume those
        modes with stale second-moment estimates. See
        :meth:`~neurom.training.base.PGDTrainer.make_optimizer`.

        Args:
            stage_index (int): Index of the stage about to run.
        """
        if stage_index > 0:
            self.decomposition.add_mode()
        for mode in range(self.decomposition.n_modes_truncated):
            self.decomposition.unfreeze_mode(mode)
        self.make_optimizer()

    def should_add_stage(self, stage_index):
        """Stop at capacity or when a new mode stops paying for itself.

        Capacity is checked first, and before ``enrichment_criterion``, so a run
        that satisfies both on the same call reports ``"capacity"``; ``add_mode``
        raises once every mode is active. Stage 0 always runs, since it adds
        nothing and merely trains the already-active modes.

        Args:
            stage_index (int): Index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """
        if stage_index == 0:
            return True
        if self.decomposition.n_modes_truncated >= self.decomposition.n_modes_max:
            self.history.stop_reason = "capacity"
            return False
        reason = self.enrichment_criterion.stop_reason(self.history.stages)
        if reason:
            self.history.stop_reason = reason
            return False
        return True

    def on_stage_end(self, record):
        """Record the newest mode's size and the worst duplication anywhere.

        ``max_correlation`` is taken over **every pair** of active modes, not
        just the newest against its predecessors as in the greedy strategy. That
        is the diagnostic this strategy actually needs: with nothing frozen, two
        modes added at different stages can drift together long after either was
        introduced, and a newest-mode-only measure would not see it.

        A tiny gain together with ``max_correlation`` near 1 means the extra
        rank bought nothing -- the decomposition is degenerate and the stage
        merely re-expressed a mode it already had.

        Args:
            record (StageRecord): The stage that just finished; ``diagnostics``
                is filled in place with ``"amplitude"`` and ``"max_correlation"``.
        """
        newest = self.decomposition.n_modes_truncated - 1
        record.diagnostics["amplitude"] = diagnostics.amplitude(
            diagnostics.monom_values(self.decomposition, newest)
        )
        record.diagnostics["max_correlation"] = diagnostics.max_pairwise_correlation(
            self.decomposition
        )
