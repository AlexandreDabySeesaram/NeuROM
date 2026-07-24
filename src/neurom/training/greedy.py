"""Purely greedy PGD training."""

import math

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

        Caveat: these use raw nodal vectors, not the quadrature-weighted L2
        inner product, so this is not the energy-norm correlation. Cheap (no
        forward pass) and good enough to *spot* duplication, not to quantify it.

        If a stage diverged, ``current``'s norms are NaN and the per-pair
        ``correlation`` propagates that NaN; it is recorded as NaN rather than
        silently swallowed. ``max(0.0, nan)`` would return ``0.0`` in Python
        (``max``/``min`` prefer the first argument on a NaN comparison), which
        would report perfect orthogonality exactly when the state is garbage
        -- the opposite of what the diagnostic is for.

        Args:
            record (StageRecord): The stage that just finished; ``diagnostics``
                is filled in place with ``"amplitude"`` and ``"max_correlation"``.
        """
        current_mode = self.decomposition.n_modes_truncated - 1
        current = self._monom_values(current_mode)

        amplitude = 1.0
        for values in current:
            amplitude *= float(values.norm())

        max_correlation = 0.0
        for earlier_mode in range(current_mode):
            earlier = self._monom_values(earlier_mode)
            correlation = 1.0
            for a, b in zip(current, earlier):
                norms = a.norm() * b.norm()
                if norms == 0.0:
                    correlation = 0.0
                    break
                correlation *= float((a * b).sum() / norms)
            correlation = abs(correlation)
            if math.isnan(correlation) or math.isnan(max_correlation):
                max_correlation = math.nan
            else:
                max_correlation = max(max_correlation, correlation)

        record.diagnostics["amplitude"] = amplitude
        record.diagnostics["max_correlation"] = max_correlation

    def _monom_values(self, mode):
        """Full nodal values of every monom of ``mode`` (constrained DOFs included).

        ``full_values()`` is correct here because the space axis's Dirichlet
        values are homogeneous (zero): every mode's constrained DOFs are zero
        too, so including them does not bias the correlation. With
        **inhomogeneous** Dirichlet data every mode would share the same
        nonzero constant component on those DOFs, and ``max_correlation``
        would be biased upward regardless of how different the free DOFs are.

        Args:
            mode (int): Index of the mode whose monoms to read.

        Returns:
            list[torch.Tensor]: One detached tensor per monom/axis of ``mode``.
        """
        return [
            field.full_values().detach() for field in self.decomposition.monoms[mode]
        ]
