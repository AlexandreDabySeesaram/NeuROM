"""Progress reporting for training runs.

Injected into the trainer rather than printed from it, so the loop stays silent
by default (tests, batch scripts) and a run in a terminal can show a bar without
either of them knowing about the other.

No dependency on tqdm: the library's runtime dependencies are torch, numpy and
matplotlib, and a stage bar is thirty lines. :class:`ProgressBar` writes to
``sys.stderr`` so a redirected stdout still carries only the results.
"""

import shutil
import sys
import time


class ProgressReporter:
    """No-op reporter, and the interface the others implement.

    The default in :class:`~neurom.training.base.PGDTrainer`: training that
    nobody is watching should print nothing.
    """

    def stage_start(self, stage_index, budget=None, label=None):
        """A stage is about to run.

        Args:
            stage_index (int): Index of the stage within the run.
            budget (int, optional): Iterations the stage criterion allows, if it
                can say -- see :meth:`~neurom.training.criteria.StageCriterion.budget`.
                None when the stage's length is not known in advance.
            label (str, optional): What the stage trains, from
                :meth:`~neurom.training.base.PGDTrainer.stage_label`. None when
                the strategy's stages are all alike.
        """

    def update(self, iteration, loss):
        """One iteration finished.

        Args:
            iteration (int): 1-based index of the iteration within the stage.
            loss (float): The loss it reported (pre-update, see
                :meth:`~neurom.training.base.PGDTrainer.step`).
        """

    def stage_end(self, record):
        """A stage finished.

        Args:
            record (StageRecord): The stage that just ran, diagnostics included.
        """

    def close(self):
        """The run finished; release whatever the reporter holds."""


class ProgressBar(ProgressReporter):
    """A one-line-per-stage terminal bar, redrawn in place.

    Shows the iteration count against the stage budget, the current energy and
    the elapsed time; on ``stage_end`` the line is finalised with the stage's
    stop reason so the finished stages stay readable above the running one::

        stage 2 |=========------| 180/600  E=-2.017e+11  12.4s
        stage 2 |===============| 203  converged  E=-2.017e+11  14.0s

    A strategy that implements
    :meth:`~neurom.training.base.PGDTrainer.stage_label` gets it beside the
    index, so a schedule with several kinds of stage per mode is readable live
    rather than only in the final table::

        stage 3 corr |====-----------| 41/600  E=-2.017e+11   3.1s

    With no ``budget`` (a criterion that cannot say how long it will run) the
    bar degrades to a spinner-less counter rather than lying about the fraction.

    Args:
        stream (IO, optional): Where to write. Defaults to ``sys.stderr``.
        width (int, optional): Bar width in characters. Defaults to a share of
            the terminal width, floored at 10.
        every (float): Minimum seconds between redraws. Redrawing on every
            iteration of a cheap stage costs more than the iteration.
    """

    def __init__(self, stream=None, width=None, every=0.1):
        self.stream = stream if stream is not None else sys.stderr
        if width is None:
            width = max(10, min(30, shutil.get_terminal_size((80, 24)).columns // 3))
        self.width = width
        self.every = every
        self.stage_index = None
        self.label = None
        self.budget = None
        self.started = None
        self.last_drawn = 0.0

    def stage_start(self, stage_index, budget=None, label=None):
        self.stage_index = stage_index
        self.label = label
        self.budget = budget
        self.started = time.monotonic()
        self.last_drawn = 0.0

    def update(self, iteration, loss):
        now = time.monotonic()
        # Always draw the first iteration -- otherwise a stage that ends inside
        # the throttle window (a short one, or a diverged one) shows nothing at
        # all and the run looks hung.
        if iteration > 1 and now - self.last_drawn < self.every:
            return
        self.last_drawn = now
        self._draw(self._body(iteration, loss))

    def stage_end(self, record):
        self._draw(
            self._body(record.n_iter, record.energy, filled=True)
            + f"  {record.stop_reason}"
        )
        self.stream.write("\n")
        self.stream.flush()

    def close(self):
        self.stream.flush()

    def _body(self, iteration, loss, filled=False):
        elapsed = time.monotonic() - self.started
        if self.budget:
            fraction = 1.0 if filled else min(1.0, iteration / self.budget)
            done = int(round(fraction * self.width))
            bar = "|" + "=" * done + " " * (self.width - done) + "|"
            counter = f"{iteration}/{self.budget}" if not filled else f"{iteration}"
        else:
            bar = ""
            counter = str(iteration)
        label = f" {self.label}" if self.label else ""
        return (
            f"stage {self.stage_index}{label} {bar} {counter:>9}"
            f"  E={loss: .4e}  {elapsed:5.1f}s"
        )

    def _draw(self, text):
        # \r + pad to the previous length: a shorter line must not leave the
        # tail of the longer one behind it.
        self.stream.write("\r" + text.ljust(self.width + 48))
        self.stream.flush()
