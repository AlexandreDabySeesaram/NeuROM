"""Purely greedy PGD training."""

from neurom.training.base import PGDTrainer


class GreedyTrainer(PGDTrainer):
    """Train one mode at a time, freezing every earlier mode permanently.

    One stage is one mode. Stage 0 trains whatever modes the decomposition was
    built active (respecting ``n_modes_ini``); every later stage freezes
    everything and activates one new mode.

    This is the classical progressive-Galerkin PGD: earlier modes are never
    revisited, so each stage minimises the energy over the residual left by its
    predecessors.

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
        """
        if stage_index > 0:
            self.decomposition.freeze_all()
            self.decomposition.add_mode()
        self.make_optimizer()

    def should_add_stage(self, stage_index):
        """Stop at capacity or when a new mode stops paying for itself.

        Capacity is checked here because ``add_mode()`` raises once every mode
        is active.
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
        """Record how large the new mode is and how much it duplicates earlier ones.

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
            max_correlation = max(max_correlation, abs(correlation))

        record.diagnostics["amplitude"] = amplitude
        record.diagnostics["max_correlation"] = max_correlation

    def _monom_values(self, mode):
        """Full nodal values of every monom of ``mode`` (constrained DOFs included)."""
        return [field.full_values().detach() for field in self.decomposition.monoms[mode]]
