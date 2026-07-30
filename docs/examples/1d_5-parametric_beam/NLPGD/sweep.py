"""Hyperparameter sweep for the 5-parametric beam **non-linear** PGD.

The twin of ``../PGD/sweep.py``, driving
``1d_5-parametric_beam_deflection_NLPGD.py`` instead. Its ledger is separate
(see :data:`LEDGER`) because the configs carry two knobs the CP ones
do not -- ``exponent_set`` and ``max_power`` -- so the content hashes, and hence
the row identities, are not comparable across the two directories. Compare the
``overall_error`` column, not the ``config_id``.

Runs a queue of ``RunConfig``s and maintains a persistent JSONL *ledger* mapping
each config to its result. Editing ``CONFIGS`` (the run queue) never loses past
results: rows are keyed by the config's content hash, so removing a config
leaves its ledger row intact and changing a knob writes a new row beside the old.

Run the queue::

    python docs/examples/1d_5-parametric_beam/NLPGD/sweep.py

The ledger path is a runner argument, so a bigger (non-parameter) change can be
sent to a fresh file.
"""

import dataclasses
import datetime as _dt
import importlib.util
import json
import os
import subprocess
import time as _time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

#: Default ledger for every helper here. The screening grid writes to its own
#: file: its rows are the first ones trained under a fixed budget with the
#: enrichment criterion disabled, so they are comparable to each other and *not*
#: to the exploratory rows in ``sweep_results.jsonl``, which stopped at whatever
#: rank ``RelativeGain`` happened to allow. Pass ``ledger=`` explicitly to reach
#: the old file.
LEDGER = "sweep_results_grid.jsonl"

_EXAMPLE = None


def _example():
    """The sibling example module, loaded by path and memoised.

    Memoised so the whole runner shares one module instance -- both so repeated
    ``run_sweep`` calls do not re-exec the script, and so a test can stub
    ``relative_errors`` on the exact object the runner reads.
    """
    global _EXAMPLE
    if _EXAMPLE is None:
        path = HERE / "1d_5-parametric_beam_deflection_NLPGD.py"
        spec = importlib.util.spec_from_file_location("beam5nl_example", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXAMPLE = module
    return _EXAMPLE


def load_ledger(path):
    """Read every row of the JSONL ledger; empty list if the file is absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def upsert_row(path, row):
    """Insert ``row``, or replace the existing one with the same ``config_id``.

    Rewrites the whole file (write-temp-then-replace) so a config re-run updates
    its row in place instead of appending a duplicate.
    """
    path = Path(path)
    rows = load_ledger(path)
    replaced = False
    for i, existing in enumerate(rows):
        if existing.get("config_id") == row["config_id"]:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def delete_row(path, config_id):
    """Drop the row with ``config_id`` from the ledger; return True if removed.

    Same write-temp-then-replace as :func:`upsert_row`. Handy to prune a stale
    experiment without hand-editing the JSONL.
    """
    path = Path(path)
    rows = load_ledger(path)
    kept = [r for r in rows if r.get("config_id") != config_id]
    if len(kept) == len(rows):
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for r in kept:
            handle.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    return True


def energy_bands(rows, tol=0.20):
    """Plausible final-energy interval per mesh, from the ledger's own rows.

    The energy is the quantity training minimises, so a run that lands far from
    where comparable runs land is suspect *without needing the FEM reference*:
    far above means the expansion never got going, far below means it drove the
    energy past the true minimum, which a correct minimiser cannot do -- it is
    the signature of a coefficient/gauge blowup.

    The centre is the **median** over rows at that mesh, not the mean: the runs
    this is meant to catch are exactly the ones that would drag a mean out to
    meet them. It is therefore only as good as the ledger it reads -- a mesh
    whose rows are mostly bad reports a bad band, and one with fewer than
    ``3`` finite rows reports none rather than a band drawn through noise.

    Args:
        rows (list[dict]): Ledger rows.
        tol (float): Half-width as a fraction of the median.

    Returns:
        dict: ``n_nodes -> (lo, hi)``, for the meshes that have enough rows.
    """
    by_mesh = {}
    for r in rows:
        energy = r["result"].get("final_energy")
        # NaN fails every comparison, so it would silently land "in band";
        # excluded here and flagged as out-of-band by `_energy_flag`.
        if energy is None or energy != energy:
            continue
        by_mesh.setdefault(r["config"].get("n_nodes"), []).append(energy)
    bands = {}
    for mesh, energies in by_mesh.items():
        if len(energies) < 3:
            continue
        energies.sort()
        middle = len(energies) // 2
        median = (
            energies[middle]
            if len(energies) % 2
            else 0.5 * (energies[middle - 1] + energies[middle])
        )
        half = abs(median) * tol
        bands[mesh] = (median - half, median + half)
    return bands


def _energy_flag(row, bands):
    """``"!"`` if the row's energy is missing, NaN, or outside its mesh's band."""
    energy = row["result"].get("final_energy")
    if energy is None or energy != energy:
        return "!"
    band = bands.get(row["config"].get("n_nodes"))
    if band is None:
        return "?"
    lo, hi = band
    return " " if lo <= energy <= hi else "!"


def show_ledger(path, energy_tol=0.20):
    """Print every ledger row, sorted by overall error (best first).

    The ``E`` column marks each row against :func:`energy_bands`: blank for a
    row in its mesh's band, ``!`` outside it (or NaN), ``?`` when the mesh has
    too few rows to draw one.
    """
    rows = load_ledger(path)
    rows.sort(key=lambda r: r["result"]["overall_error"])
    name_width = max([len("name")] + [len(r["config"]["name"]) for r in rows])

    def _clip(name):
        if len(name) <= name_width:
            return name
        return name[: name_width - 1] + "…"

    columns = [
        ("name", name_width, "<"),
        ("config_id", 9, ">"),
        ("overall", 10, ">"),
        ("worst", 10, ">"),
        ("energy", 12, ">"),
        ("E", 1, "<"),
        ("modes", 5, ">"),
        ("iters", 6, ">"),
        ("secs", 7, ">"),
        ("seed", 6, ">"),
        ("I", 14, "<"),
        ("stop", 10, "<"),
    ]
    header = "  ".join(f"{title:{align}{width}}" for title, width, align in columns)
    print(header)
    print("-" * len(header))
    bands = energy_bands(rows, tol=energy_tol)
    for r in rows:
        res, cfg = r["result"], r["config"]
        # Wall clock of the training run. Rows written before it was recorded
        # show "-"; it is machine- and load-dependent, so read it as a cost
        # signal next to `iters`, never as a benchmark between rows measured
        # on different days.
        secs = r.get("meta", {}).get("train_seconds")
        secs_str = f"{secs:7.1f}" if secs is not None else f"{'-':>7}"
        # Older rows predate the seed_amplitude knob; fall back so a mixed ledger
        # still prints (the row is stale anyway and will be rerun).
        seed = cfg.get("seed_amplitude")
        seed_str = f"{seed:6.3g}" if seed is not None else f"{'-':>6}"
        # The last stage's energy -- the quantity the training actually
        # minimises, where `overall` is measured against the FEM reference.
        # Comparable across rows only at equal mesh: the energy is an integral
        # over the discretisation, so a refined `n_nodes` moves it on its own.
        # It is also *not* comparable to the ../PGD ledger's column for the same
        # reason the ids are not -- same mesh there, but check before reading.
        energy = res.get("final_energy")
        energy_str = f"{energy:12.4e}" if energy is not None else f"{'-':>12}"
        # The exponent set is the knob this ledger has that the CP one does not.
        # A CP-baseline row still carries it (the decomposition is built with it)
        # but never uses it, so it is shown parenthesised for those strategies.
        exponents = f"{cfg.get('exponent_set', '-')}{cfg.get('max_power', '')}"
        if cfg.get("strategy") in ("greedy", "simultaneous"):
            exponents = f"({exponents})"
        print(
            "  ".join(
                [
                    f"{_clip(cfg['name']):<{name_width}}",
                    f"{r['config_id']:>9}",
                    f"{res['overall_error']:10.3e}",
                    f"{res['worst_point_error']:10.3e}",
                    energy_str,
                    _energy_flag(r, bands),
                    f"{res['n_modes']:5d}",
                    f"{res['total_iters']:6d}",
                    secs_str,
                    seed_str,
                    f"{exponents:<14}",
                    f"{res['stop_reason']:<10}",
                ]
            )
        )


def rank_curve(pgd, reference=None):
    """Error at every rank ``1..n_modes_truncated``, from one trained run.

    Greedy enrichment is nested -- mode ``m`` was trained against the residual
    of the ``m`` before it -- so a rank-N decomposition already contains its own
    rank-1..N approximations. Scoring each of them costs an evaluation, not a
    training run, which is why the sweep can drop the usual "screen at low rank,
    confirm at high rank" two-phase design: every row reports where its own
    accuracy-vs-rank knee sits.

    Read it, not just the final number: two configs that tie at rank 10 are not
    equivalent if one got there at rank 6, and a curve still falling at the last
    rank means the run was rank-limited rather than converged.

    Args:
        pgd: a trained decomposition exposing ``truncated``.
        reference (dict, optional): a loaded reference bundle, passed through to
            the example's ``relative_errors``. Load it once and pass it in --
            the default re-reads it from disk on every call, once per rank.

    Returns:
        list[dict]: one ``{"rank", "overall", "worst"}`` per rank, ascending.
    """
    ex = _example()
    curve = []
    for rank in range(1, int(pgd.n_modes_truncated) + 1):
        with torch.no_grad(), pgd.truncated(rank):
            errors = ex.relative_errors(pgd, reference)
        per_point = errors["per_point"]
        curve.append(
            {
                "rank": rank,
                "overall": errors["overall"],
                "worst": per_point[max(per_point, key=per_point.get)],
            }
        )
    return curve


def show_rank_curve(config_id, ledger=None):
    """Print one row's stored error-vs-rank curve, with the gain per added mode.

    Reads the ledger, not a checkpoint: the curve is recorded at training time
    by :func:`extract_result`. Rows written before that show nothing -- rerun
    them, or call :func:`rank_curve` on a :func:`load_run` result.

    Args:
        config_id (str): the row's ``config_id``, e.g. from :func:`show_ledger`.
        ledger (str or Path, optional): JSONL file the row lives in; defaults to
            :data:`LEDGER`.
    """
    ledger = Path(ledger if ledger is not None else LEDGER)
    if not ledger.is_absolute():
        ledger = HERE / ledger
    rows = load_ledger(ledger)
    row = next((r for r in rows if r["config_id"] == config_id), None)
    if row is None:
        raise KeyError(f"no row with config_id {config_id!r} in {ledger}")
    curve = row["result"].get("rank_curve")
    if not curve:
        print(f"{config_id}: no rank_curve recorded (row predates it) -- rerun it")
        return

    print(f"{row['config']['name']}  ({config_id})")
    print(f"{'rank':>4}  {'overall':>10}  {'worst':>10}  {'d(worst)':>9}")
    previous = None
    for point in curve:
        # Gain per added mode, the quantity that says where the curve flattens.
        # Blank on the first rank, which has nothing to improve on.
        delta = f"{previous - point['worst']:9.2e}" if previous is not None else " " * 9
        print(
            f"{point['rank']:4d}  {point['overall']:10.3e}  "
            f"{point['worst']:10.3e}  {delta}"
        )
        previous = point["worst"]


def extract_result(problem, errors, reference=None):
    """Assemble the ledger ``result`` block from a finished run and its errors."""
    history = problem.history
    per_point = errors["per_point"]
    worst_label = max(per_point, key=per_point.get)
    stages = [
        {
            "stage": r.stage,
            "n_iter": r.n_iter,
            "stop_reason": r.stop_reason,
            "energy": r.energy,
            "gain": r.gain,
            "amplitude": r.diagnostics.get("amplitude"),
            "max_correlation": r.diagnostics.get("max_correlation"),
            # The two NL-only diagnostics. `kind` names what the stage trained
            # (see SCHEDULES); `coefficient_norm` is the L1 norm of its mode's
            # coefficient row -- the only recorded quantity that looks at C at
            # all, and hence the only way to tell from the ledger alone whether
            # a run released anything. Rows written before this line lack both.
            "kind": r.diagnostics.get("kind", "cp"),
            "coefficient_norm": r.diagnostics.get("coefficient_norm", 0.0),
        }
        for r in history.stages
    ]
    return {
        "overall_error": errors["overall"],
        # Error at every intermediate rank, from this same trained run -- see
        # `rank_curve`. The last entry restates `overall_error`/`worst_point_error`.
        "rank_curve": rank_curve(problem.pgd, reference),
        "worst_point_error": per_point[worst_label],
        "worst_point_label": worst_label,
        "per_point_error": dict(per_point),
        "final_energy": history.stages[-1].energy,
        "n_modes": problem.pgd.n_modes_truncated,
        "n_stages": len(history.stages),
        "stop_reason": history.stop_reason,
        "total_iters": sum(r.n_iter for r in history.stages),
        "stages": stages,
    }


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return None


def run_sweep(configs, ledger=None, retrain=False,
              verbose=True, plot=False, checkpoint_dir=None):
    """Train every uncached config in ``configs`` and record it in ``ledger``.

    A config already present in the ledger (by ``config_id``) is skipped unless
    ``retrain`` -- the seconds-vs-minutes shortcut. Each trained config upserts
    its row and saves a checkpoint keyed by id. Returns the whole ledger.

    Args:
        configs (list[RunConfig]): the run queue.
        ledger (str or Path, optional): JSONL file; relative paths resolve
            next to this script. Defaults to :data:`LEDGER`. A bigger,
            non-parameter change can target a fresh path.
        retrain (bool): train (and overwrite the row/checkpoint) even if cached.
        verbose (bool): print per-config progress and the training tables.
        plot (bool): draw the example's figures for each trained config.
        checkpoint_dir (str or Path, optional): where ``nlpgd5_sweep_<id>.pt``
            live; defaults to the example's ``PARAM_SWEEP_DIR``.

    Returns:
        list[dict]: every ledger row after the run.
    """
    ex = _example()
    ledger = Path(ledger if ledger is not None else LEDGER)
    if not ledger.is_absolute():
        ledger = HERE / ledger
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else ex.PARAM_SWEEP_DIR

    known = {r["config_id"] for r in load_ledger(ledger)}
    for cfg in configs:
        cid = ex.config_id(cfg)
        if cid in known and not retrain:
            if verbose:
                print(f"cached  : {cfg.name} ({cid})")
            continue

        trainer_cls = ex.STRATEGIES[cfg.strategy]
        checkpoint = checkpoint_dir / f"nlpgd5_sweep_{cid}.pt"
        started = _time.monotonic()
        if verbose:
            # Announced BEFORE training, not just on the "recorded" line after:
            # a config runs for minutes, and the example's own tables scroll past
            # with nothing in them naming which config produced them.
            print(f"training: {cfg.name} ({cid}) "
                  f"strategy={cfg.strategy} "
                  f"I={cfg.exponent_set}{cfg.max_power} "
                  f"modes<={cfg.n_modes_max} min_iter={cfg.min_iter} "
                  f"lr={cfg.lr} lr_C={cfg.coefficient_lr}")
        problem = ex.main(
            verbose=verbose,
            plot=plot,
            trainer_cls=trainer_cls,
            config=cfg,
            checkpoint=checkpoint,
            retrain=retrain,
        )
        reference = ex.load_reference_bundle()
        errors = ex.relative_errors(problem.pgd, reference)
        row = {
            "config_id": cid,
            "config": dataclasses.asdict(cfg),
            "result": extract_result(problem, errors, reference),
            "meta": {
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                # Wall clock, this machine, including the error evaluation. A
                # cost figure to read beside `total_iters`, not a benchmark:
                # it is not comparable across machines or across a busy one.
                "train_seconds": round(_time.monotonic() - started, 1),
                "checkpoint": checkpoint.name,
                "git_commit": _git_commit(),
                "ledger_schema": 1,
            },
        }
        upsert_row(ledger, row)
        known.add(cid)
        if verbose:
            print(f"recorded: {cfg.name} ({cid}) "
                  f"overall={row['result']['overall_error']:.3e} "
                  f"in {row['meta']['train_seconds']:.1f}s")

    return load_ledger(ledger)


def load_run(config_id, ledger=None, checkpoint_dir=None):
    """Reload a sweep row's trained ``Problem`` from its checkpoint.

    Looks up ``config_id`` in ``ledger``, rebuilds its ``RunConfig`` from the
    stored row, and loads the matching checkpoint -- no retraining, seconds not
    minutes, same mechanism ``run_sweep`` relies on.

    Args:
        config_id (str): the row's ``config_id``, e.g. from ``show_ledger``.
        ledger (str or Path, optional): JSONL file; relative paths resolve
            next to this script. Defaults to :data:`LEDGER`.
        checkpoint_dir (str or Path, optional): where ``nlpgd5_sweep_<id>.pt``
            live; defaults to the example's ``PARAM_SWEEP_DIR``, matching
            ``run_sweep``.

    Returns:
        Problem: with ``.pgd`` and ``.history`` filled from the checkpoint.
    """
    ex = _example()
    ledger = Path(ledger if ledger is not None else LEDGER)
    if not ledger.is_absolute():
        ledger = HERE / ledger
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else ex.PARAM_SWEEP_DIR

    rows = load_ledger(ledger)
    row = next((r for r in rows if r["config_id"] == config_id), None)
    if row is None:
        raise KeyError(f"no row with config_id {config_id!r} in {ledger}")

    cfg = ex.RunConfig(**row["config"])
    trainer_cls = ex.STRATEGIES[cfg.strategy]
    checkpoint = checkpoint_dir / row["meta"]["checkpoint"]
    return ex.main(
        verbose=False,
        plot=False,
        trainer_cls=trainer_cls,
        config=cfg,
        checkpoint=checkpoint,
        retrain=False,
    )


def show_run(config_id, ledger=None, checkpoint_dir=None):
    """Reprint one row's full training log -- the per-stage table and ``C``.

    The same output the run itself printed, regenerated from its checkpoint
    rather than from a retrain, because the ``TrainingHistory`` rides along in
    the checkpoint. Use it when the ledger's one-line summary is not enough:
    the per-stage table is where ``max corr`` (are the modes copies of each
    other?) and ``|C|`` (did the correction move at all?) live.

    Args:
        config_id (str): the row's ``config_id``, e.g. from :func:`show_ledger`.
        ledger (str or Path, optional): JSONL file the row lives in;
            defaults to :data:`LEDGER`.
        checkpoint_dir (str or Path, optional): passed through to
            :func:`load_run`.

    Returns:
        Problem: the reloaded run, for further poking.
    """
    ex = _example()
    kwargs = {"ledger": ledger}
    if checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = checkpoint_dir
    problem = load_run(config_id, **kwargs)
    ex._report(problem, problem.history, verbose=True, plot=False)
    return problem


def plot_losses(config_id, ledger=None, checkpoint_dir=None,
                 save_path=None):
    """Redraw the convergence curve (energy vs. iteration) for one sweep row.

    Thin wrapper around the example's ``plot_convergence``, fed by a checkpoint
    reload instead of a live training run.

    Args:
        config_id (str): the row's ``config_id``.
        ledger (str or Path, optional): JSONL file the row lives in;
            defaults to :data:`LEDGER`.
        checkpoint_dir (str or Path, optional): passed through to
            :func:`load_run`.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "nlpgd5_sweep_<config_id>_convergence.png"``.
    """
    ex = _example()
    kwargs = {"ledger": ledger}
    if checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = checkpoint_dir
    problem = load_run(config_id, **kwargs)
    ex.plot_convergence(
        problem.history,
        save_path=save_path or ex.PLOT_DIR / f"nlpgd5_sweep_{config_id}_convergence.png",
    )


def plot_extremes(config_id, ledger=None, checkpoint_dir=None,
                    labels=None, save_path=None):
    """Redraw the two-hardest-points comparison for one sweep row.

    Thin wrapper around the example's ``plot_solution``, fed by a checkpoint
    reload instead of a live training run.

    Args:
        config_id (str): the row's ``config_id``.
        ledger (str or Path, optional): JSONL file the row lives in;
            defaults to :data:`LEDGER`.
        checkpoint_dir (str or Path, optional): passed through to
            :func:`load_run`.
        labels (list[str], optional): which reference points to draw; defaults
            to the reference bundle's highlighted pair.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "nlpgd5_sweep_<config_id>_vs_reference.png"``.

    Returns:
        dict: the ``relative_errors`` result, as ``plot_solution`` returns.
    """
    ex = _example()
    kwargs = {"ledger": ledger}
    if checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = checkpoint_dir
    problem = load_run(config_id, **kwargs)
    return ex.plot_solution(
        problem.pgd,
        labels=labels,
        save_path=save_path or ex.PLOT_DIR / f"nlpgd5_sweep_{config_id}_vs_reference.png",
    )


# --- Screening grid -------------------------------------------------------
#
# A full factorial over the four knobs that are still open, run at *fixed
# budget* so the rows are comparable to each other. The exploratory rows in
# `sweep_results.jsonl` are not: they stopped at ranks between 5 and 10 and at
# iteration counts between 649 and 4626, so a difference between two of them
# says as much about the budget they got as about the knobs they carry.
#
# What is held fixed, and why:
#
#   n_modes_max     10 for every cell, and rank is NOT a factor in the design.
#                   Greedy enrichment is nested, so one rank-10 run contains its
#                   own rank-1..10 solutions and `rank_curve` scores all of them
#                   from the finished checkpoint. Training a cheap rank-5 screen
#                   would have measured the knobs at a rank below the knee --
#                   worst-case error falls steeply from 5 to ~7 modes and only
#                   flattens after -- i.e. in a regime nothing will be shipped
#                   in. This costs about 2x a rank-5 screen and replaces both
#                   it and the rank-10 confirmation phase.
#   enrichment_tol  DISABLED (see SCREEN_NO_ENRICHMENT_STOP). Every screen row
#                   must reach the same rank or the error column conflates
#                   "worse knobs" with "stopped sooner".
#   min_iter        350, clearing the documented ~300-per-stage floor below
#                   which a fresh mode never launches. Below it the screen would
#                   manufacture a failure that is not the knobs' fault.
#   stage_tol       1e-5, expected inert: with max_iter=400 every stage should
#                   stop on `max_iter`. If a row reports otherwise its budget
#                   was not the others' -- check before comparing it.
#
# Cost is NOT uniform across the cells: it grows with `1 + |I|`, and the
# `total_degree` levels are the expensive end (a rank-3 td7 row cost 602s where
# a rank-5 uniform2 row cost 16s). Cheapest index sets are queued first so the
# early feedback arrives early.

#: Disables :class:`RelativeGain` without removing it from the pipeline. It
#: stops when ``(previous - current) / denom < tol``; a negative tol therefore
#: fires only if the energy *rose* by more than the denominator, i.e. on a
#: blowup, never on an ordinary flat stage. Enrichment is then governed solely
#: by ``n_modes_max``, and every row stops on ``capacity``.
#:
#: This matters because the criterion has no patience: it compares stage n to
#: stage n-1 only, so one flat stage ends the run. Three of the exploratory
#: rows finished at 5 modes that way and reported ``converged`` while sitting
#: at 40x the error of the rows that were allowed to continue.
SCREEN_NO_ENRICHMENT_STOP = -1.0

#: ``(exponent_set, max_power)`` levels, cheapest first. One factor, not two:
#: the integer *is* ``max_power``, and it means "largest power" for ``uniform``
#: but "largest total degree" for ``total_degree`` (which needs > 5 here, five
#: being the number of axes). ``uniform5`` is kept although it has already
#: NaN'd once -- a divergence boundary is a result, and the point is to learn
#: whether it is the index set or the learning rates that put it there.
SCREEN_INDEX_SETS = [
    ("uniform", 2),
    ("uniform", 3),
    ("uniform", 5),
    ("total_degree", 6),
    ("total_degree", 7),
]

#: Crossed with :data:`SCREEN_LRS` deliberately. Read alone the ledger's
#: coefficient rates look ragged (1e-6 fine, 1e-5 blows up, 1e-4 fine, 1e-3
#: blows up), which is the signature of a confound with ``lr`` rather than of a
#: genuinely non-monotone optimum -- every blowup sits on a row that also raised
#: ``lr``. Crossing the two separates them.
SCREEN_COEFFICIENT_LRS = [1e-6, 1e-5, 1e-4]

#: Not frozen at an intermediate value, because in the exploratory ledger ``lr``
#: is perfectly confounded with the index set: every ``uniform`` row ran 1e-2
#: and every ``total_degree`` row ran 1e-1. Freezing it would hand one family a
#: rate tuned for the other and blame the difference on the index set.
SCREEN_LRS = [1e-2, 1e-1]


def screen_configs(ex, strategy, n_modes_max=10, max_iter=400, min_iter=350):
    """The full factorial for one strategy, as ``RunConfig``s.

    Names are *derived* from the fields rather than typed, so a name can never
    drift from the config it labels -- the exploratory ledger has rows called
    ``...-uniform3`` carrying ``uniform5`` and ``...-total_degree6`` carrying
    ``total_degree7``, from hand-edited copies. Identity is the content hash
    either way, so those rows are valid; their labels are not.

    Args:
        ex (module): the loaded example module.
        strategy (str): key into ``ex.STRATEGIES``.
        n_modes_max (int): rank. 10, and not a cheaper screening rank: the
            run is nested, so `rank_curve` recovers every lower rank from it
            for free -- there is nothing to be gained by training at 5 and
            guessing whether the ordering transfers.
        max_iter (int): per **stage**, not per run.
        min_iter (int): per stage; keep >= 350, see the module comment.

    Returns:
        list: ``len(SCREEN_INDEX_SETS) * len(SCREEN_COEFFICIENT_LRS) *
        len(SCREEN_LRS)`` configs.
    """
    configs = []
    for exponent_set, max_power in SCREEN_INDEX_SETS:
        for lr in SCREEN_LRS:
            for coefficient_lr in SCREEN_COEFFICIENT_LRS:
                configs.append(
                    ex.RunConfig(
                        name=(
                            f"screen-{strategy}-{exponent_set}{max_power}"
                            f"-lr{lr:g}-clr{coefficient_lr:g}"
                            f"-r{n_modes_max}"
                        ),
                        strategy=strategy,
                        exponent_set=exponent_set,
                        max_power=max_power,
                        lr=lr,
                        coefficient_lr=coefficient_lr,
                        n_modes_max=n_modes_max,
                        max_iter=max_iter,
                        min_iter=min_iter,
                        stage_tol=1e-5,
                        enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,
                    )
                )
    return configs


def _configs():
    """The run queue -- edit this to enter configurations.

    Removing an entry leaves its ledger row intact; changing a knob writes a new
    row beside the old one. Built lazily so the example module loads only when
    the sweep actually runs.

    Currently: the ``joint`` half of the screening grid, 30 rows. The ``refine``
    half is the same call with ``strategy="refine"`` and goes in once these have
    landed -- ``joint`` is the arm with 13 exploratory rows behind it, so it is
    the one that can be sanity-checked against something.
    """
    ex = _example()
    return screen_configs(ex, "joint")


if __name__ == "__main__":
    import sys

    retrain = "--retrain" in sys.argv[1:]
    run_sweep(_configs(), retrain=retrain)
    show_ledger(HERE / LEDGER)
