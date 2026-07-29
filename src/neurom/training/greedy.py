"""Purely greedy PGD training."""

from neurom.training import diagnostics
from neurom.training.base import PGDTrainer


class GreedyTrainer(PGDTrainer):
    """Train one mode at a time, freezing every earlier mode permanently.

    One stage is one mode. Stage 0 trains whatever modes the decomposition was
    built active (respecting ``n_modes_ini``); every later stage freezes
    everything and activates one new mode.

    This is the classical progressive-Galerkin PGD: earlier modes are never
    revisited, so each stage minimises the energy over the residual left by its
    predecessors.

    ``n_modes_ini > 1`` buys nothing under the current constant seeding:
    every initially-active mode starts from the same ``Axis.init_values``
    seed and sees the same gradient throughout stage 0's joint training, so
    they stay parallel forever (measured: `max_correlation = 1.0`, amplitudes
    matching to 4 significant digits). Only ``n_modes_ini=1`` avoids wasting
    parameters this way.

    Args:
        model (NeuROMModel): Model whose ``decomposition`` is a ``CPPGD``.
        optimizer_factory, stage_criterion, enrichment_criterion: See
            :class:`~neurom.training.base.PGDTrainer`.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.decomposition = model.decomposition

    def prepare_stage(self, stage_index):
        """Freeze everything and activate one new mode, then build an optimizer.

        ``freeze_all()`` before ``add_mode()`` -- rather than freezing only the
        previous mode -- states "purely greedy" outright. Freezing the previous
        mode alone reaches the same state only because every mode happens to be
        frozen in turn.

        Stage 0 adds nothing: the decomposition already has ``n_modes_ini``
        active modes, and they are what stage 0 trains.

        Args:
            stage_index (int): Index of the stage about to run.
        """
        if stage_index > 0:
            self.decomposition.freeze_all()
            self.decomposition.add_mode()
        # Fix the scale gauge before the optimizer is built: renormalising
        # rescales parameters, which would invalidate any carried-over optimizer
        # state (Adam moments). No-op unless the decomposition implements one.
        self.decomposition.renormalise()
        self.make_optimizer()

    def should_add_stage(self, stage_index):
        """Stop at capacity or when a new mode stops paying for itself.

        Capacity is checked here because ``add_mode()`` raises once every mode
        is active. It is checked *before* ``enrichment_criterion``, so a run
        that would satisfy both on the same call reports ``"capacity"``, not
        the enrichment criterion's reason. Stage 0 always returns True
        unconditionally -- so ``MaxStages(0)``, which would otherwise stop
        before any stage runs, still gets exactly one stage in, since stage 0
        trains whatever modes are already active rather than adding one.

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
        """Record how large the last active mode is and how much it duplicates earlier ones.

        Every mode is seeded with the same ``Axis.init_values``, so successive
        greedy steps could in principle converge to the same mode. Freezing is
        what should prevent it -- a later mode descends a different residual --
        but that is a hope, not a guarantee, so measure it:

        * ``amplitude``: ``prod_k ||w_m^k||``, the mode's overall size.
        * ``max_correlation``: over earlier modes, the largest
          ``prod_k <w_i^k, w_j^k> / (||w_i^k|| ||w_j^k||)`` -- the normalised
          inner product of the two rank-1 tensors.

        A tiny gain together with ``max_correlation`` near 1 is the greedy
        sequence rediscovering a mode it already has.

        Only the newest mode is tested: every earlier one is frozen, so no pair
        of them can drift together after the fact. A strategy that keeps old
        modes trainable needs ``max_pairwise_correlation`` instead -- see
        :class:`~neurom.training.simultaneous.SimultaneousTrainer`.

        See :mod:`neurom.training.diagnostics` for what these measure and for
        the nodal-vector caveat. NaN from a diverged stage is recorded as NaN
        rather than silently swallowed.

        Args:
            record (StageRecord): The stage that just finished; ``diagnostics``
                is filled in place with ``"amplitude"`` and ``"max_correlation"``.
        """
        current_mode = self.decomposition.n_modes_truncated - 1
        record.diagnostics["amplitude"] = diagnostics.amplitude(
            diagnostics.monom_values(self.decomposition, current_mode)
        )
        record.diagnostics["max_correlation"] = diagnostics.max_correlation(
            self.decomposition, current_mode
        )
