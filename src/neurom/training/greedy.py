"""Purely greedy PGD training."""

from neurom.training.base import PGDTrainer


class GreedyTrainer(PGDTrainer):
    """Train one mode at a time, freezing every earlier mode permanently.

    One stage is one mode. Stage 0 trains the modes the decomposition was built
    with; every later stage freezes everything and adds one new mode. Earlier
    modes are never revisited, so each stage minimises the energy over the
    residual left by its predecessors.

    Args:
        model (NeuROMModel): Model whose ``decomposition`` is a ``CPPGD``.
        phases, enrichment_criterion: See
            :class:`~neurom.training.base.PGDTrainer`.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.decomposition = model.decomposition

    def prepare_stage(self, stage_index):
        """Freeze everything, add one mode and register it.

        Stage 0 adds nothing: it trains the modes already in the decomposition.

        Args:
            stage_index (int): Index of the stage about to run.
        """
        if stage_index > 0:
            self.decomposition.freeze_all()
            self.decomposition.add_mode()
            self.decomposition.register_into(self.model.field_layout)

    def should_add_stage(self, stage_index):
        """Run stage 0 unconditionally, then ask the enrichment criterion.

        Args:
            stage_index (int): Index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """
        if stage_index == 0:
            return True
        reason = self.enrichment_criterion.stop_reason(self.history.stages)
        if reason:
            self.history.stop_reason = reason
            return False
        return True
