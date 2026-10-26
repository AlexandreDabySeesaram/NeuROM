"""Progress reporting for training runs.

Injected into the trainer rather than printed from it, so the loop stays silent
by default (tests, batch scripts) and a run in a notebook or a terminal can
report as it goes without either of them knowing about the other.

Adapted from ``progress.py`` on ``origin/nl_pgd``: the hooks follow the current
loop (stages made of :class:`~neurom.training.base.Phase`), and the reporter
prints whole lines, which read the same in a terminal and in a notebook output
(no ``\\r`` redraw).
"""

import sys
import time


class ProgressReporter:
    """No-op reporter, and the interface the others implement.

    The default in :class:`~neurom.training.base.PGDTrainer`: training that
    nobody is watching should print nothing.
    """

    def stage_start(self, stage_index):
        """A stage is about to run its phases.

        Args:
            stage_index (int): Index of the stage within the run.
        """

    def phase_start(self, name):
        """A phase of the current stage is about to run.

        Args:
            name (str): The phase's name (``Phase.name``).
        """

    def update(self, iteration, loss):
        """One iteration of the current phase finished.

        Args:
            iteration (int): 1-based index of the iteration within the phase.
            loss (float): The loss it reported (pre-update, see
                :meth:`~neurom.training.base.PGDTrainer.step`).
        """

    def phase_end(self, name, n_iter, reason, loss):
        """A phase finished.

        Args:
            name (str): The phase's name.
            n_iter (int): Iterations it ran.
            reason (str): Why it stopped (``"converged"``, ``"max_iter"``,
                ``"diverged"``, ...).
            loss (float or None): Its last loss, None if it ran no iteration.
        """

    def stage_end(self, record):
        """A stage finished.

        Args:
            record (StageRecord): The stage that just ran.
        """

    def close(self, history):
        """The run finished.

        Args:
            history (TrainingHistory): The whole run.
        """


def _duration(seconds):
    """`12.3 s` below a minute, `3 min 12 s` above."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes} min {seconds:02d} s"


class ProgressLog(ProgressReporter):
    """One line per event, e.g.::

        mode 0: start
          adam   it   500  loss = -3.4102e+07  4.1 s
          adam   done: 2339 it, converged, loss = -3.4352e+07, 12.3 s
          lbfgs  done: 10 it, max_iter, loss = -3.4352e+07, 0.9 s
        mode 0: done in 2349 it, max_iter, energy = -3.4352e+07, 13.2 s
        run done: 15 modes, n_stages, 3 min 12 s

    A stage is called a mode: one stage adds one mode for both the greedy and
    the simultaneous strategies.

    Args:
        every (int or None): Print the loss every ``every`` iterations of a
            phase; None prints only the start and end lines.
        stream (IO, optional): Where to write. Defaults to ``sys.stdout``.
    """

    def __init__(self, every=500, stream=None):
        self.every = every
        self.stream = stream
        self.run_started = None
        self.stage_started = None
        self.phase_started = None
        self.phase_name = None

    def _print(self, text):
        print(text, file=self.stream if self.stream is not None else sys.stdout, flush=True)

    def stage_start(self, stage_index):
        now = time.monotonic()
        if self.run_started is None:
            self.run_started = now
        self.stage_started = now
        self._print(f"mode {stage_index}: start")

    def phase_start(self, name):
        self.phase_name = name
        self.phase_started = time.monotonic()

    def update(self, iteration, loss):
        if self.every and iteration % self.every == 0:
            elapsed = time.monotonic() - self.phase_started
            self._print(
                f"  {self.phase_name:<6} it {iteration:>5}  loss = {loss: .4e}  {_duration(elapsed)}"
            )

    def phase_end(self, name, n_iter, reason, loss):
        elapsed = time.monotonic() - self.phase_started
        last = f"loss = {loss: .4e}" if loss is not None else "no iteration"
        self._print(f"  {name:<6} done: {n_iter} it, {reason}, {last}, {_duration(elapsed)}")

    def stage_end(self, record):
        elapsed = time.monotonic() - self.stage_started
        self._print(
            f"mode {record.stage}: done in {record.n_iter} it, {record.stop_reason}, "
            f"energy = {record.energy: .4e}, {_duration(elapsed)}"
        )

    def close(self, history):
        elapsed = time.monotonic() - self.run_started if self.run_started is not None else 0.0
        self._print(
            f"run done: {len(history.stages)} modes, {history.stop_reason}, {_duration(elapsed)}"
        )
