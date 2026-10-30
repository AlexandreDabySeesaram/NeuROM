"""PGD training that keeps every mode trainable, after Algorithm 1 of
Daby-Seesaram et al. (2026), Part II."""

import torch

from neurom.training.base import PGDTrainer


class SimultaneousTrainer(PGDTrainer):
    """Add one mode per stage, keep training **all** the modes, and keep the
    optimizer from one stage to the next.

    This is the training of the FENNI-PGD, §3.2.1 and Algorithm 1 of
    Daby-Seesaram, Škardová & Genet (2026), Comput. Mech. 77(2) 521–546. The
    paper calls it *greedy* because modes come one at a time; it is not
    :class:`~neurom.training.greedy.GreedyTrainer`, which freezes the earlier
    modes. Here nothing is ever frozen: stage ``m`` minimises the energy over
    the whole rank-``m+1`` decomposition, from the previous stage's modes.

    The optimizer is built once, at the first stage, then each new mode is
    added to it as a parameter group
    (:meth:`~neurom.neurom_model.NeuROMModel.add_mode_to_optimizer`), so the
    state of the earlier modes (e.g. Adam's moments) carries over, as with
    ``AddMode2Optimizer`` in Algorithm 1.

    Algorithm 1 reads::

        SimultaneousTrainer(
            model,
            phases=[Phase(adam, Stagnation(tol=1e-4), "adam")],
            enrichment_criterion=AnyOf(Usefulness(tol=1e-4), MaxStages(n_max)),
            final_phases=[Phase(lbfgs, Stagnation(patience=0, no_improvement=None), "lbfgs")],
        )

    :class:`~neurom.training.criteria.Stagnation` ends a stage (new mode),
    :class:`~neurom.training.criteria.Usefulness` ends the run when the last
    mode did not help, and the final LBFGS is the second stage of §3.2.2. One
    difference: Algorithm 1 tests the end of the run inside the iteration loop,
    with its own patience ``max_stgn``; here it is tested between stages, so a
    useless mode stops after the stage criterion's patience. The paper reports
    the result insensitive to both patiences between 5 and 100.

    Worth expecting:

    * **Enrichment gains are not comparable to greedy's.** A stage lowers the
      energy both by adding a mode and by correcting the old ones.
    * **Modes can collapse onto one another.** Freezing is what keeps greedy's
      modes apart; two free modes can converge to the same rank-1 tensor.
    * **Stage budgets do not transfer from greedy.** A stage spends part of its
      iterations re-fitting the old modes.

    With ``keep_optimizer=False``, every phase of every stage gets a fresh
    optimizer over all the modes instead, as in
    :class:`~neurom.training.base.PGDTrainer`. Each stage still trains the whole
    decomposition, but no optimizer state carries over. This allows an LBFGS
    per stage, and several phases per stage (e.g. Adam then LBFGS), which a
    kept optimizer cannot: ``torch.optim.LBFGS`` caches its single parameter
    group, so a group added later would be silently ignored, and a kept Adam
    would resume with moments made stale by the LBFGS steps in between.

    Args:
        model (NeuROMModel): Model whose ``decomposition`` is a ``CPPGD``.
        keep_optimizer (bool): Keep the stage optimizer across stages
            (Algorithm 1), or build a fresh one for every phase.
        phases (list[Phase], optional): The phases of every stage; exactly one
            when ``keep_optimizer`` is True. See
            :class:`~neurom.training.base.PGDTrainer`.
        enrichment_criterion, progress, final_phases: See
            :class:`~neurom.training.base.PGDTrainer`. The final phases get
            fresh optimizers; give them a factory other than the stage
            phase's, or they would reuse its optimizer.

    Raises:
        ValueError: With ``keep_optimizer`` True, if there is not exactly one
            phase, or if its optimizer is an LBFGS.
    """

    def __init__(self, model, *, keep_optimizer=True, **kwargs):
        super().__init__(model, **kwargs)
        self.keep_optimizer = keep_optimizer
        if keep_optimizer and len(self.phases) != 1:
            raise ValueError(
                "SimultaneousTrainer keeps one optimizer across stages, so it takes "
                f"exactly one phase per stage (got {len(self.phases)}); put a "
                "polishing optimizer in final_phases, or pass keep_optimizer=False."
            )
        self.decomposition = model.decomposition
        self.kept_optimizer = None
        self.n_modes_in_optimizer = 0

    def prepare_stage(self, stage_index):
        """Add one mode (from stage 1 on), register it, and unfreeze every mode.

        The explicit ``unfreeze_all()`` is not redundant: the decomposition may
        arrive frozen (e.g. from a greedy run), so the stage states its own
        requirement.

        Args:
            stage_index (int): Index of the stage about to run.
        """
        if stage_index > 0:
            self.decomposition.add_mode()
            self.decomposition.register_into(self.model.field_layout)
        self.decomposition.unfreeze_all()

    def make_optimizer(self, optimizer_factory):
        """Build the stage optimizer once, then add the new modes to it.

        Any other factory (the final phases), or any factory when
        ``keep_optimizer`` is False, gets a fresh optimizer.

        Args:
            optimizer_factory (Callable): ``params -> Optimizer``.

        Returns:
            torch.optim.Optimizer: The optimizer, also stored on
            ``self.optimizer``.

        Raises:
            ValueError: If the kept stage optimizer is an LBFGS.
        """
        kept = optimizer_factory is self.phases[0].optimizer_factory
        if not (self.keep_optimizer and kept):
            return super().make_optimizer(optimizer_factory)
        if self.kept_optimizer is None:
            optimizer = super().make_optimizer(optimizer_factory)
            if isinstance(optimizer, torch.optim.LBFGS):
                raise ValueError(
                    "LBFGS cannot take the parameters of a new mode; use it in "
                    "final_phases, or pass keep_optimizer=False."
                )
            self.kept_optimizer = optimizer
        else:
            for m in range(self.n_modes_in_optimizer, self.decomposition.n_modes):
                self.model.add_mode_to_optimizer(self.kept_optimizer, m)
        self.n_modes_in_optimizer = self.decomposition.n_modes
        self.optimizer = self.kept_optimizer
        return self.optimizer
