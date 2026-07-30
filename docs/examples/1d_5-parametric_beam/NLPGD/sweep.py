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

#: Default ledger for every helper here. The new-strategies campaign writes to
#: its own file: its rows carry knobs (``n_linear_modes``, ``leading_coefficient``,
#: ``pin_space_exponent``) that no earlier row has, so their content hashes -- and
#: hence their identities -- are new anyway, and keeping them apart stops the
#: median in :func:`energy_bands` from being drawn through two different designs.
#:
#: The earlier files are still reachable by passing ``ledger=`` explicitly:
#: ``sweep_results_grid.jsonl`` (the fixed-budget screening grid) and
#: ``sweep_results.jsonl`` (the exploratory rows, which stopped at whatever rank
#: ``RelativeGain`` happened to allow and are not budget-comparable).
LEDGER = "sweep_results_new_strategies.jsonl"

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


def _mesh_key(n_nodes):
    """Hashable identity of a config's mesh, for grouping rows by it.

    ``RunConfig.n_nodes`` is a per-axis dict (or ``None`` for the example's
    default), and a dict cannot key the grouping in :func:`energy_bands` -- which
    is why setting a mesh explicitly used to raise ``unhashable type: 'dict'``
    there. Sorted, so two rows that wrote the same mesh with different key order
    still group together.

    Anything else (``None``, or the bare int the single-axis sweeps use) is
    already hashable and passes through unchanged.
    """
    if isinstance(n_nodes, dict):
        return tuple(sorted(n_nodes.items()))
    return n_nodes


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
        by_mesh.setdefault(_mesh_key(r["config"].get("n_nodes")), []).append(energy)
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
    band = bands.get(_mesh_key(row["config"].get("n_nodes")))
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
        ("l/c/g", 5, "<"),
        ("max|c|", 10, ">"),
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
        if cfg.get("pin_space_exponent"):
            # The space exponent pinned to 1: same builder, fewer/other rows, so
            # the bare `uniform3` label would name two different sets.
            exponents += "|x=1"
        if cfg.get("strategy") in ("greedy", "simultaneous"):
            exponents = f"({exponents})"
        # The three knobs that cut across the strategies: the linear-phase
        # length, whether the non-linear modes carry their own leading
        # coefficient, and whether the gauge is fixed at all. "-" for rows
        # written before a knob existed; those rows ran with l=0, no `c` and
        # `renormalise` on, which is what the defaults reproduce. `g` is shown
        # for the gauge fix being ON, so a bare "0/-/-" is the one row where the
        # flat directions were left unfixed -- the thing you want to spot.
        linear = cfg.get("n_linear_modes")
        phase = "-" if linear is None else str(linear)
        phase += "/c" if cfg.get("leading_coefficient") else "/-"
        phase += "/g" if cfg.get("renormalise", True) else "/-"
        # Largest final |c| over the modes -- a one-number blowup detector, since
        # `c` is where `renormalise` parks each mode's whole amplitude. "-" when
        # the decomposition pins `c` at 1, and for rows written before it was
        # recorded. Read the trajectory with `show_leading_coefficients`: this
        # column says how big, never whether it settled.
        leading = res.get("leading_coefficients")
        max_c = f"{max(abs(v) for v in leading):10.3e}" if leading else f"{'-':>10}"
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
                    f"{phase:<5}",
                    max_c,
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


def show_leading_coefficients(config_id, ledger=None):
    """Print one row's leading coefficients ``c_m``, per stage -- did they settle?

    One column per mode, one line per stage, so a ``c`` that is still climbing at
    the last stage is visible as a column that never flattens. The final line
    restates the ledger's ``max|c|`` column.

    Two things to know before reading it, or the table will mislead:

    * **``c`` moves on stages that did not train it.** ``fix_gauge`` runs over
      every active mode at the start of every stage, and it parks each mode's
      whole amplitude in that mode's ``c``. So an earlier mode's column keeps
      changing long after its own stage ended, and it does so even when ``c`` is
      frozen -- the rescale is an in-place write under ``no_grad``, and
      ``requires_grad=False`` blocks gradients, not writes. "``c`` changed" is
      therefore *not* evidence that the optimizer touched it.
    * **Under ``renormalise`` the natural size of ``c`` is the mode's amplitude**,
      which is ~1e5 on this problem, not ~1. A large ``c`` is the gauge working,
      not a blowup. What would be a blowup is a column growing without settling
      while the energy stops falling.

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

    stages = row["result"].get("stages", [])
    tracked = [s for s in stages if s.get("leading_coefficients")]
    if not tracked:
        # Two different causes, and the distinction matters: the run may have
        # pinned c at 1 (nothing to show), or it may predate the diagnostic
        # (rerun it). The config says which.
        if row["config"].get("leading_coefficient"):
            print(f"{config_id}: leading coefficients not recorded (row predates "
                  "the diagnostic) -- rerun it with --retrain")
        else:
            print(f"{config_id}: ran with leading_coefficient=False, so every "
                  "c is pinned at 1 and there is nothing to show")
        return

    n_modes = max(len(s["leading_coefficients"]) for s in tracked)
    print(f"{row['config']['name']}  ({config_id})")
    print(f"{'stage':>5}  {'mode':>4}  {'kind':>7}  "
          + "  ".join(f"{'c' + str(m):>11}" for m in range(n_modes)))
    for s in tracked:
        values = s["leading_coefficients"]
        cells = [
            f"{values[m]:11.4e}" if m < len(values) else f"{'':>11}"
            for m in range(n_modes)
        ]
        # `mode` is the mode this stage trained; with a linear phase it is no
        # longer the stage index, which is why it is stored rather than derived.
        print(f"{s['stage']:5d}  {s.get('mode', s['stage']):4d}  "
              f"{s.get('kind', 'cp'):>7}  " + "  ".join(cells))


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
            # The active mode's leading coefficient, and every active mode's, at
            # the end of this stage. Both absent under a decomposition that pins
            # `c` at 1 (every run without `leading_coefficient`), and absent from
            # rows written before this line -- `show_leading_coefficients` says
            # so rather than printing zeros.
            "leading_coefficient": r.diagnostics.get("leading_coefficient"),
            "leading_coefficients": r.diagnostics.get("leading_coefficients"),
        }
        for r in history.stages
    ]
    pgd = problem.pgd
    leading = None
    if getattr(pgd, "has_leading_coefficients", False):
        leading = [
            float(pgd.leading_coefficients[m].detach())
            for m in range(int(pgd.n_modes_truncated))
        ]
    return {
        # Final `c` per mode, or None when the decomposition pins it at 1. Read
        # from the decomposition rather than from the last stage record so it is
        # the value actually stored in the checkpoint.
        "leading_coefficients": leading,
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


# --- Screening grid (DORMANT) ---------------------------------------------
#
# Commented out, not deleted: its rows are already in `sweep_results_grid.jsonl`
# and that ledger stays readable (`show_ledger(HERE / "sweep_results_grid.jsonl")`).
# The design notes below are what makes those rows interpretable, so they stay
# here too. Uncomment the constants and `screen_configs` to extend the grid --
# e.g. to a strategy it has not covered.
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

# #: ``(exponent_set, max_power)`` levels, cheapest first. One factor, not two:
# #: the integer *is* ``max_power``, and it means "largest power" for ``uniform``
# #: but "largest total degree" for ``total_degree`` (which needs > 5 here, five
# #: being the number of axes). ``uniform5`` is kept although it has already
# #: NaN'd once -- a divergence boundary is a result, and the point is to learn
# #: whether it is the index set or the learning rates that put it there.
# SCREEN_INDEX_SETS = [
#     ("uniform", 2),
#     ("uniform", 3),
#     ("uniform", 5),
#     ("total_degree", 6),
#     ("total_degree", 7),
# ]
#
# #: Crossed with :data:`SCREEN_LRS` deliberately. Read alone the ledger's
# #: coefficient rates look ragged (1e-6 fine, 1e-5 blows up, 1e-4 fine, 1e-3
# #: blows up), which is the signature of a confound with ``lr`` rather than of a
# #: genuinely non-monotone optimum -- every blowup sits on a row that also raised
# #: ``lr``. Crossing the two separates them.
# SCREEN_COEFFICIENT_LRS = [1e-6, 1e-5, 1e-4]
#
# #: Not frozen at an intermediate value, because in the exploratory ledger ``lr``
# #: is perfectly confounded with the index set: every ``uniform`` row ran 1e-2
# #: and every ``total_degree`` row ran 1e-1. Freezing it would hand one family a
# #: rate tuned for the other and blame the difference on the index set.
# SCREEN_LRS = [1e-2, 1e-1]
#
#
# def screen_configs(ex, strategy, n_modes_max=10, max_iter=400, min_iter=350):
#     """The full factorial for one strategy, as ``RunConfig``s.
#
#     Names are *derived* from the fields rather than typed, so a name can never
#     drift from the config it labels -- the exploratory ledger has rows called
#     ``...-uniform3`` carrying ``uniform5`` and ``...-total_degree6`` carrying
#     ``total_degree7``, from hand-edited copies. Identity is the content hash
#     either way, so those rows are valid; their labels are not.
#
#     Args:
#         ex (module): the loaded example module.
#         strategy (str): key into ``ex.STRATEGIES``.
#         n_modes_max (int): rank. 10, and not a cheaper screening rank: the
#             run is nested, so `rank_curve` recovers every lower rank from it
#             for free -- there is nothing to be gained by training at 5 and
#             guessing whether the ordering transfers.
#         max_iter (int): per **stage**, not per run.
#         min_iter (int): per stage; keep >= 350, see the module comment.
#
#     Returns:
#         list: ``len(SCREEN_INDEX_SETS) * len(SCREEN_COEFFICIENT_LRS) *
#         len(SCREEN_LRS)`` configs.
#     """
#     configs = []
#     for exponent_set, max_power in SCREEN_INDEX_SETS:
#         for lr in SCREEN_LRS:
#             for coefficient_lr in SCREEN_COEFFICIENT_LRS:
#                 configs.append(
#                     ex.RunConfig(
#                         name=(
#                             f"screen-{strategy}-{exponent_set}{max_power}"
#                             f"-lr{lr:g}-clr{coefficient_lr:g}"
#                             f"-r{n_modes_max}"
#                         ),
#                         strategy=strategy,
#                         exponent_set=exponent_set,
#                         max_power=max_power,
#                         lr=lr,
#                         coefficient_lr=coefficient_lr,
#                         n_modes_max=n_modes_max,
#                         max_iter=max_iter,
#                         min_iter=min_iter,
#                         stage_tol=1e-5,
#                         enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,
#                     )
#                 )
#     return configs


#: The linear-phase lengths to try. 0 is the control -- the schedule from mode 0,
#: i.e. what every existing row ran -- so a difference at l > 0 is attributable.
SCREEN_LINEAR_MODES = [0, 2]


def strategy_screen_configs(ex, n_modes_max=10, max_iter=400, min_iter=350):
    """The new strategies' arm of the screen, at the best-known index set.

    Deliberately **not** a full factorial with the index-set/lr grid of
    :func:`screen_configs`. The two new knobs are what is being screened here;
    crossing them with everything else would be ~10x the rows for a comparison
    that ``joint``'s arm already answers. The index set and rates are pinned to
    one setting so the l/c/pin contrasts are read against a fixed background.

    Six rows, three contrasts, each one knob apart from a neighbour:

    ===========================  ==================================================
    ``joint``, l = 0             the control -- identical to the existing arm
    ``joint``, l = 2             does a linear phase help at all?
    ``joint``, l = 2, ``c``      does the leading coefficient help on top?
    ``support``, l = 0           frozen support alone
    ``support``, l = 2           the fusion the plan settles on
    ``support``, l = 2, pinned   ... with the space exponent held at 1
    ===========================  ==================================================

    Args:
        ex (module): the loaded example module.
        n_modes_max (int): rank; the run is nested, so `rank_curve` recovers
            every lower rank from it for free.
        max_iter, min_iter (int): per **stage**, not per run. A two-stage
            schedule therefore spends 2x per mode -- which `support` does and
            `joint` does not, so read `iters` alongside `modes`.

    Returns:
        list: six ``RunConfig``s.
    """
    def cfg(strategy, linear=0, leading=False, pinned=False):
        label = f"{strategy}-l{linear}" + ("-c" if leading else "") + ("-pin" if pinned else "")
        return ex.RunConfig(
            name=f"strat-{label}-r{n_modes_max}",
            strategy=strategy,
            n_linear_modes=linear,
            leading_coefficient=leading,
            pin_space_exponent=pinned,
            exponent_set=STRATEGY_SCREEN_INDEX_SET[0],
            max_power=STRATEGY_SCREEN_INDEX_SET[1],
            lr=STRATEGY_SCREEN_LR,
            coefficient_lr=STRATEGY_SCREEN_COEFFICIENT_LR,
            n_modes_max=n_modes_max,
            max_iter=max_iter,
            min_iter=min_iter,
            stage_tol=1e-5,
            enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,
        )

    return [
        cfg("joint"),
        cfg("joint", linear=2),
        cfg("joint", linear=2, leading=True),
        cfg("support"),
        cfg("support", linear=2),
        cfg("support", linear=2, pinned=True),
    ]


#: Background for the strategy screen, held fixed so the l/c/pin contrasts are
#: read against one setting rather than confounded with the index set. Chosen as
#: the cheapest of SCREEN_INDEX_SETS; **not** claimed to be the best -- the
#: index-set screen has not been read yet.
STRATEGY_SCREEN_INDEX_SET = ("uniform", 3)
STRATEGY_SCREEN_LR = 1e-1
STRATEGY_SCREEN_COEFFICIENT_LR = 1e-3


def _configs():
    """The run queue -- edit this to enter configurations.

    Removing an entry leaves its ledger row intact; changing a knob writes a new
    row beside the old one. Built lazily so the example module loads only when
    the sweep actually runs.

    Currently **one base config, every field of ``RunConfig`` written out**,
    including the ones that merely restate a default. That is deliberate: the
    ledger stores ``dataclasses.asdict(cfg)``, so a row records the value of a
    knob whether or not it was typed here -- but the *queue* only shows what was
    typed, and a default that moves later would silently change what "the base
    config" meant. Spelling every field out makes this file, on its own, the
    complete statement of what was run.

    To vary a knob: copy the block, change the one field, change ``name``.
    Identity is the content hash, so the two rows land side by side.

    :func:`strategy_screen_configs` still builds the six-row l/c/pin screen;
    ``return strategy_screen_configs(ex)`` to queue it instead.
    """
    ex = _example()
    return [
        # ex.RunConfig(
        #     name="l3-lead_coeffTrue-totaldeg6-coefflr1e-3-r7",
        #     # --- what is trained, and in what order ---------------------------
        #     strategy="joint",           
        #     n_linear_modes=3,           # pure-CP modes before the NL schedule
        #     leading_coefficient=True,  # release c_i on the NL modes
        #     pin_space_exponent=False,   # pin_axis(I, space, 1)
        #     renormalise=False,           # fix the scale gauge each stage

        #     # --- the polynomial correction ------------------------------------
        #     exponent_set="total_degree",     # "uniform" | "total_degree"
        #     max_power=6,                # largest power (uniform) / total degree
        #     # --- rank and mesh -------------------------------------------------
        #     n_modes_max=7,             # per mode (rank), not per stage
        #     n_nodes=None,               # None -> ex.DEFAULT_N_NODES
        #     seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
        #     # --- stage schedule (all per STAGE, not per run) -------------------
        #     min_iter=100,               # >= ~300 or a fresh mode never launches
        #     max_iter=600,
        #     stage_tol=1e-5,             # inert at max_iter=400; check if a row
        #     stage_floor=1.0,            #   reports otherwise before comparing it
        #     window=20,
        #     linear_stage_tol=1e-3,      # None -> reuse stage_tol
        #     # --- enrichment ----------------------------------------------------
        #     enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
        #     enrichment_floor=1.0,
        #     # --- optimiser ------------------------------------------------------
        #     lr=1e-1,
        #     coefficient_lr=1e-4,
        # ),
        ex.RunConfig(
            name="l3-lead_coeffTrue-stratSupportPinSpace-totaldeg6-r6-renorm",
            # --- what is trained, and in what order ---------------------------
            strategy="support",           
            n_linear_modes=3,           # pure-CP modes before the NL schedule
            leading_coefficient=True,  # release c_i on the NL modes
            pin_space_exponent=True,   # pin_axis(I, space, 1)
            renormalise=True,           # fix the scale gauge each stage

            # --- the polynomial correction ------------------------------------
            exponent_set="total_degree",     # "uniform" | "total_degree"
            max_power=6,                # largest power (uniform) / total degree
            # --- rank and mesh -------------------------------------------------
            n_modes_max=6,             # per mode (rank), not per stage
            n_nodes=None,               # None -> ex.DEFAULT_N_NODES
            seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
            # --- stage schedule (all per STAGE, not per run) -----x`--------------
            min_iter=150,               # >= ~300 or a fresh mode never launches
            max_iter=600,
            stage_tol=1e-5,             # inert at max_iter=400; check if a row
            stage_floor=1.0,            #   reports otherwise before comparing it
            window=20,
            linear_stage_tol=1e-3,      # None -> reuse stage_tol
            # --- enrichment ----------------------------------------------------
            enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
            enrichment_floor=1.0,
            # --- optimiser ------------------------------------------------------
            lr=1e-1,
            coefficient_lr=1e-4,
        ),
        # ex.RunConfig(
        #     name="l3-lead_coeffTrue-uniform2-r10-renorm",
        #     # --- what is trained, and in what order ---------------------------
        #     strategy="joint",
        #     n_linear_modes=3,           # pure-CP modes before the NL schedule
        #     leading_coefficient=True,   # release c_i on the NL modes
        #     pin_space_exponent=False,   # pin_axis(I, space, 1)
        #     renormalise=True,           # THE knob under test
        #     # --- the polynomial correction ------------------------------------
        #     exponent_set="uniform",     # "uniform" | "total_degree"
        #     max_power=2,                # largest power (uniform) / total degree
        #     # --- rank and mesh -------------------------------------------------
        #     n_modes_max=10,             # per mode (rank), not per stage
        #     n_nodes=None,               # None -> ex.DEFAULT_N_NODES
        #     seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
        #     # --- stage schedule (all per STAGE, not per run) -------------------
        #     min_iter=300,               # >= ~300 or a fresh mode never launches
        #     max_iter=600,
        #     stage_tol=1e-5,             # inert at max_iter=400; check if a row
        #     stage_floor=1.0,            #   reports otherwise before comparing it
        #     window=20,
        #     linear_stage_tol=1e-3,      # None -> reuse stage_tol
        #     # --- enrichment ----------------------------------------------------
        #     enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
        #     enrichment_floor=1.0,
        #     # --- optimiser ------------------------------------------------------
        #     lr=1e-1,
        #     coefficient_lr=1e-4,
        # ),
        # ex.RunConfig(
        #     name="l3-lead_coeffTrue-uniform3-r6-renorm",
        #     # --- what is trained, and in what order ---------------------------
        #     strategy="joint",
        #     n_linear_modes=3,           # pure-CP modes before the NL schedule
        #     leading_coefficient=True,   # release c_i on the NL modes
        #     pin_space_exponent=False,   # pin_axis(I, space, 1)
        #     renormalise=True,           # THE knob under test
        #     # --- the polynomial correction ------------------------------------
        #     exponent_set="uniform",     # "uniform" | "total_degree"
        #     max_power=3,                # largest power (uniform) / total degree
        #     # --- rank and mesh -------------------------------------------------
        #     n_modes_max=6,              # per mode (rank), not per stage
        #     n_nodes=None,               # None -> ex.DEFAULT_N_NODES
        #     seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
        #     # --- stage schedule (all per STAGE, not per run) -------------------
        #     min_iter=300,               # >= ~300 or a fresh mode never launches
        #     max_iter=600,
        #     stage_tol=1e-5,             # inert at max_iter=400; check if a row
        #     stage_floor=1.0,            #   reports otherwise before comparing it
        #     window=20,
        #     linear_stage_tol=1e-3,      # None -> reuse stage_tol
        #     # --- enrichment ----------------------------------------------------
        #     enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
        #     enrichment_floor=1.0,
        #     # --- optimiser ------------------------------------------------------
        #     lr=1e-1,
        #     coefficient_lr=1e-4,
        # ),
        # Single-knob controls against the two `-renorm` rows above (e1850154
        # and 944186d3): same everything, `orthogonal_corrections` on.
        #
        # `renormalise` removed the *leverage* -- uniform3 stopped diverging,
        # 209% -> 8.81% overall -- but not the *redundancy*: the (3,3,3,3,3) row
        # still holds 99.5% of mode 3 and 100% of mode 5, because with the monoms
        # free it spans the same rank-1 set as the leading term and can replace
        # it at no cost in energy. Deflation removes that overlap outright.
        #
        # What to read, in this order:
        #   1. the term shares in the final recap -- does the leading term keep a
        #      non-trivial share, or does one correction still take the mode?
        #   2. `overall`, against 4.232e-02 (uniform2) and 8.811e-02 (uniform3).
        # A row that only rebalances the shares without moving the error says the
        # redundancy was real but harmless, which is itself worth knowing.
        #
        # Expect these to be slower: 1 + 2|I| terms instead of 1 + |I|, and the
        # energy's double loop is quadratic -- 2.8x at |I| = 2, 2.25x at |I| = 1.
        ex.RunConfig(
            name="l3-lead_coeffTrue-uniform2-r10-renorm-orth",
            # --- what is trained, and in what order ---------------------------
            strategy="joint",
            n_linear_modes=3,           # pure-CP modes before the NL schedule
            leading_coefficient=True,   # release c_i on the NL modes
            pin_space_exponent=False,   # pin_axis(I, space, 1)
            renormalise=True,           # fix the scale gauge each stage
            orthogonal_corrections=True,  # THE knob under test
            # --- the polynomial correction ------------------------------------
            exponent_set="uniform",     # "uniform" | "total_degree"
            max_power=2,                # largest power (uniform) / total degree
            # --- rank and mesh -------------------------------------------------
            n_modes_max=10,             # per mode (rank), not per stage
            n_nodes=None,               # None -> ex.DEFAULT_N_NODES
            seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
            # --- stage schedule (all per STAGE, not per run) -------------------
            min_iter=300,               # >= ~300 or a fresh mode never launches
            max_iter=600,
            stage_tol=1e-5,             # inert at max_iter=400; check if a row
            stage_floor=1.0,            #   reports otherwise before comparing it
            window=20,
            linear_stage_tol=1e-3,      # None -> reuse stage_tol
            # --- enrichment ----------------------------------------------------
            enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
            enrichment_floor=1.0,
            # --- optimiser ------------------------------------------------------
            lr=1e-1,
            coefficient_lr=1e-4,
        ),
        ex.RunConfig(
            name="l3-lead_coeffTrue-uniform3-r6-renorm-orth",
            # --- what is trained, and in what order ---------------------------
            strategy="joint",
            n_linear_modes=3,           # pure-CP modes before the NL schedule
            leading_coefficient=True,   # release c_i on the NL modes
            pin_space_exponent=False,   # pin_axis(I, space, 1)
            renormalise=True,           # fix the scale gauge each stage
            orthogonal_corrections=True,  # THE knob under test
            # --- the polynomial correction ------------------------------------
            exponent_set="uniform",     # "uniform" | "total_degree"
            max_power=3,                # largest power (uniform) / total degree
            # --- rank and mesh -------------------------------------------------
            n_modes_max=6,              # per mode (rank), not per stage
            n_nodes=None,               # None -> ex.DEFAULT_N_NODES
            seed_amplitude=0.05,        # must stay > 0, see CPPGD.add_mode
            # --- stage schedule (all per STAGE, not per run) -------------------
            min_iter=300,               # >= ~300 or a fresh mode never launches
            max_iter=600,
            stage_tol=1e-5,             # inert at max_iter=400; check if a row
            stage_floor=1.0,            #   reports otherwise before comparing it
            window=20,
            linear_stage_tol=1e-3,      # None -> reuse stage_tol
            # --- enrichment ----------------------------------------------------
            enrichment_tol=SCREEN_NO_ENRICHMENT_STOP,  # disabled; stop on rank
            enrichment_floor=1.0,
            # --- optimiser ------------------------------------------------------
            lr=1e-1,
            coefficient_lr=1e-4,
        ),
    ]


if __name__ == "__main__":
    import sys

    retrain = "--retrain" in sys.argv[1:]
    run_sweep(_configs(), retrain=retrain)
    show_ledger(HERE / LEDGER)
