"""Base class for PGD training strategies."""

import math
from abc import ABC, abstractmethod

import torch

from neurom.training.criteria import RelativeChange, RelativeGain
from neurom.training.history import StageRecord, TrainingHistory
from neurom.training.progress import ProgressReporter


def _default_optimizer(params):
    return torch.optim.Adam(params, lr=0.1)


class PGDTrainer(ABC):
    """Template for a PGD training strategy.

    The loop is ``enrich`` -> ``stage`` -> ``step``: a run is a sequence of
    *stages*, each of which iterates until its criterion fires.

    **The base knows nothing about modes.** It never calls ``add_mode`` or
    ``mode_parameters``, and it never assumes a stage index is a mode index.
    Everything that touches the decomposition happens in :meth:`prepare_stage`,
    and the trainable set is read off the freeze state::

        [p for p in model.parameters() if p.requires_grad]

    which names whatever is currently trainable -- a CP mode, one axis of one, or
    a global non-linear correction -- without the base having to know which. That
    is what lets the same loop drive the non-linear decompositions, whose terms
    are not all modes.

    Subclasses implement :meth:`prepare_stage` and :meth:`should_add_stage`.
    Overriding :meth:`step` gives a different inner iteration (alternating
    directions); overriding :meth:`stage` gives a different inner loop entirely.

    Args:
        model (NeuROMModel): The model to train. Must be callable with no
            arguments in training mode and expose ``loss(output)``.
        optimizer_factory (Callable, optional): ``params -> Optimizer``. Called
            once per stage, on the currently-unfrozen parameters. Defaults to
            ``Adam(lr=0.1)``.
        stage_criterion (StageCriterion, optional): When a stage is done.
            Defaults to :class:`RelativeChange`.
        enrichment_criterion (EnrichmentCriterion, optional): When to stop
            starting stages. Defaults to :class:`RelativeGain`.
        progress (ProgressReporter, optional): Watches the run. Defaults to
            :class:`~neurom.training.progress.ProgressReporter`, which does
            nothing -- training nobody is watching prints nothing. Pass a
            :class:`~neurom.training.progress.ProgressBar` for a terminal bar.
        renormalise (bool, optional): Whether :meth:`prepare_stage` fixes the
            decomposition's scale gauge at each stage boundary. Defaults to
            ``True``. It lives here rather than on the decomposition because it
            is a *training* choice: the gauge fix is an exact reparameterisation,
            so it changes the optimisation path, never the represented field.
            Set ``False`` to train the raw, degenerate parameterisation -- the
            ablation that measures what the fix buys. No-op for a decomposition
            whose ``renormalise()`` does nothing (``CPPGD``).

    Attributes:
        optimizer (torch.optim.Optimizer): The current stage's optimizer, or
            None before the first :meth:`prepare_stage`.
        history (TrainingHistory): Accumulates across ``enrich`` calls, so a run
            can be resumed.
    """

    def __init__(
        self,
        model,
        optimizer_factory=None,
        stage_criterion=None,
        enrichment_criterion=None,
        progress=None,
        renormalise=True,
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory or _default_optimizer
        self.stage_criterion = stage_criterion or RelativeChange()
        self.enrichment_criterion = enrichment_criterion or RelativeGain()
        self.progress = progress or ProgressReporter()
        self.renormalise = bool(renormalise)
        self.optimizer = None
        self.history = TrainingHistory()

    def fix_gauge(self):
        """Fix the decomposition's scale gauge, unless ``renormalise`` is off.

        Call from :meth:`prepare_stage` *before* :meth:`make_optimizer`: the
        gauge fix rescales parameters, so any optimizer state referring to them
        (Adam's moments) must be discarded rather than carried over.
        """
        if self.renormalise:
            self.model.decomposition.renormalise()

    def trainable_parameters(self):
        """The parameters the current freeze state leaves trainable.

        Returns:
            list[torch.nn.Parameter]: Every parameter of ``self.model`` with
                ``requires_grad`` set.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def make_optimizer(self):
        """Build a fresh optimizer over the currently-trainable parameters.

        A fresh optimizer per stage matters less for purely greedy training --
        a frozen parameter gets no gradient and ``Adam.step`` skips it either
        way -- than for the strategies that *unfreeze* earlier modes. Adam's
        moment buffers persist in ``optimizer.state`` indefinitely, so a reused
        optimizer would resume an unfrozen mode with moment estimates
        accumulated against the energy at a different truncation order. A fresh
        optimizer makes that impossible rather than something to remember.

        Returns:
            torch.optim.Optimizer: The newly built optimizer, also stored on
                ``self.optimizer``.

        Raises:
            RuntimeError: If nothing is trainable, which means ``prepare_stage``
                left everything frozen.
        """
        params = self.trainable_parameters()
        if not params:
            raise RuntimeError(
                "No trainable parameters: prepare_stage left everything frozen."
            )
        self.optimizer = self.optimizer_factory(params)
        return self.optimizer

    def enrich(self):
        """Run stages until the strategy or a criterion stops them.

        Resumable: the next stage index is ``len(self.history.stages)``, and the
        history accumulates, so calling this again continues from the current
        state.

        Leaves the model in training mode; call ``model.eval()`` before
        evaluating at arbitrary points.

        Returns:
            TrainingHistory: The accumulated history (also on ``self.history``).
        """
        self.model.train()
        while self.should_add_stage(len(self.history.stages)):
            stage_index = len(self.history.stages)
            self.prepare_stage(stage_index)
            record = self.stage(stage_index)
            self.on_stage_end(record)
            self.history.append(record)
            if record.diverged:
                self.history.stop_reason = "diverged"
                break
        else:
            # `while ... else`: reached only when the loop ran out of stages, not
            # when it broke on divergence -- where the parameters are NaN and
            # anything the hook does to them would pile onto the real failure.
            self.on_run_end()
        self.progress.close()
        return self.history

    def on_run_end(self):
        """Hook for whatever a strategy owes the final state. Default: nothing.

        Called once, after the last stage, and **only** when the run ended
        normally. The base has nothing to do here -- it knows nothing about
        modes -- so this exists for the subclasses that fix a gauge in
        :meth:`prepare_stage`: that is *before* a stage, so the last stage of the
        run is never followed by one, and without this hook a run ends holding
        whatever scale it happened to drift to.
        """

    def stage(self, stage_index):
        """Iterate :meth:`step` until the stage criterion fires.

        Divergence is checked twice, because ``step()`` only ever reports the
        loss *before* that iteration's update:

        - Inside the loop, on every recorded (pre-update) loss. A non-finite
          value here ends the stage immediately as diverged, which stops the
          whole run.
        - After the criterion fires and the post-loop final energy has been
          evaluated, on that final energy. This is what catches the case
          where the stage's *last* update is what blows the model up: every
          recorded loss is still finite, so the in-loop check cannot see it,
          but the state the stage actually ended in is not finite either.

        This energy multiplies five factors and passes one through a tanh, so
        blow-up is realistic, and a NaN would otherwise silently poison every
        later stage.

        Args:
            stage_index (int): Index of this stage within the run.

        Returns:
            StageRecord: The stage's losses and outcome.
        """
        record = StageRecord(stage=stage_index)
        self.progress.stage_start(
            stage_index,
            self.stage_criterion.budget(),
            label=self.stage_label(stage_index),
        )
        while True:
            loss = self.step()
            record.losses.append(loss)
            self.progress.update(record.n_iter, loss)
            if not math.isfinite(loss):
                record.diverged = True
                record.stop_reason = "diverged"
                record.final_energy = loss
                self.progress.stage_end(record)
                return record
            reason = self.stage_criterion.stop_reason(record.losses)
            if reason:
                record.stop_reason = reason
                self._record_final_energy(record)
                # step() only ever sees PRE-update losses, so the final update
                # can blow the model up without any recorded loss showing it.
                if not math.isfinite(record.final_energy):
                    record.diverged = True
                    record.stop_reason = "diverged"
                self.progress.stage_end(record)
                return record

    def step(self):
        """Run one optimizer iteration.

        Closure-based, so second-order optimizers that re-evaluate the loss
        (LBFGS) work with no branching.

        Returns:
            float: The loss **before** this iteration's update -- that is what
            ``optimizer.step(closure)`` hands back. A monotonicity check written
            without knowing this will look off by one.
        """
        return float(self.optimizer.step(self._closure()).detach())

    def _closure(self):
        """Build the closure ``optimizer.step`` calls.

        Separate from :meth:`step` so a strategy overriding ``step`` (e.g.
        alternating directions, which sweeps axis by axis) reuses it instead of
        copying the ``retain_graph`` subtlety below.

        Returns:
            Callable[[], torch.Tensor]: Zeroes the gradients, runs a forward,
            evaluates the loss, backpropagates, and returns the loss.
        """

        def closure():
            self.optimizer.zero_grad()
            output = self.model()
            loss = self.model.loss(output)
            # retain_graph=True is required, not speculative. QuadratureContext
            # builds x_phys and xi_back once at construction from the
            # requires_grad leaf _xi_ref (_compute_quad_pos), and
            # NeuROMModel.forward never calls IntegrationDomain.update_contexts(),
            # so every iteration's forward reads through that same already-built
            # subgraph. A non-retaining backward() frees it, and the second
            # iteration then fails with "Trying to backward through the graph a
            # second time". Any energy reading interpolated quantities hits this,
            # not only autograd-differentiated ones.
            #
            # The alternative fix is calling update_contexts() every forward,
            # which rebuilds the subgraph and also works, at the cost of a
            # re-map per context per iteration. Retaining is cheaper while the
            # geometry is static; revisit if meshes become trainable.
            loss.backward(retain_graph=True)
            return loss

        return closure

    def _record_final_energy(self, record):
        """Evaluate the loss of the state the stage ended in.

        One extra forward pass per stage -- negligible against hundreds of
        iterations -- and it is what makes ``StageRecord.energy`` describe the
        state the stage produced rather than the one before its last update.

        Deliberately **not** wrapped in ``torch.no_grad()``: a physics loss that
        computes spatial derivatives via ``torch.autograd.grad`` (see
        ``neurom.differential.jacobian_field``, used by the 5-parametric beam's
        energy) needs an active, grad-tracking forward pass to differentiate
        through at all -- under ``no_grad()`` every intermediate tensor loses
        its ``grad_fn`` and that inner ``autograd.grad`` call raises "element 0
        of tensors does not require grad and does not have a grad_fn" the
        instant it is evaluated, which is exactly what
        ``tests/integration/test_greedy_trainer.py`` hit here on real data (the
        stub loss in ``tests/unit/training/test_base.py`` has no such internal
        differentiation, so it never surfaced this). ``.detach()`` gets the
        same "no gradient escapes this call" outcome without disabling the
        graph the loss needs internally -- the same pattern the pre-trainer
        2-parameter example uses (``loss.detach().item()``) rather than
        ``no_grad()``.

        Args:
            record (StageRecord): The stage record to fill in, in place.
        """
        record.final_energy = float(self.model.loss(self.model()).detach())

    @abstractmethod
    def prepare_stage(self, stage_index):
        """Set the freeze state for this stage and build its optimizer.

        The strategy's main variation point, and the only place the
        decomposition is manipulated. The optimizer must be in place before
        the stage's first :meth:`step` -- usually that means calling
        :meth:`make_optimizer` here, at the end of ``prepare_stage``, but that
        is not required: a strategy that builds a different optimizer per axis
        (e.g. alternating directions) may instead arrange for one to exist by
        the time :meth:`step` -- which it also overrides -- first runs.

        Args:
            stage_index (int): Index of the stage about to run.
        """

    @abstractmethod
    def should_add_stage(self, stage_index):
        """Whether to run stage ``stage_index``.

        Implementations that stop should set ``self.history.stop_reason`` before
        returning False, so the history says why the run ended.

        Args:
            stage_index (int): Index of the stage that would run next.

        Returns:
            bool: True to run the stage, False to stop the run.
        """

    def stage_label(self, stage_index):
        """Short name for what this stage trains, for the progress reporter.

        Default None: a strategy whose stages are all alike has nothing to add
        to the index the bar already prints. A multi-stage-per-mode strategy
        overrides it so the live bar says which half of the schedule is running
        -- the same string it will later put in ``record.diagnostics``, derived
        here once rather than re-derived by the reporter from an index parity it
        does not own.

        Args:
            stage_index (int): Index of the stage about to run.

        Returns:
            str or None: The label, or None for no label.
        """
        return None

    def on_stage_end(self, record):
        """Hook called after each stage, before it enters the history.

        Default does nothing. Strategies fill ``record.diagnostics`` here.

        Args:
            record (StageRecord): The stage that just finished.
        """
