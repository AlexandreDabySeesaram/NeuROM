"""Hyperparameter sweep for the 5-parametric beam PGD.

Runs a queue of ``RunConfig``s and maintains a persistent JSONL *ledger* mapping
each config to its result. Editing ``CONFIGS`` (the run queue) never loses past
results: rows are keyed by the config's content hash, so removing a config
leaves its ledger row intact and changing a knob writes a new row beside the old.

Run the queue::

    python docs/examples/1d_5-parametric_beam_PGD/sweep.py

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

HERE = Path(__file__).resolve().parent

_EXAMPLE = None


def _example():
    """The sibling example module, loaded by path and memoised.

    Memoised so the whole runner shares one module instance -- both so repeated
    ``run_sweep`` calls do not re-exec the script, and so a test can stub
    ``relative_errors`` on the exact object the runner reads.
    """
    global _EXAMPLE
    if _EXAMPLE is None:
        path = HERE / "1d_5-parametric_beam_deflection_PGD.py"
        spec = importlib.util.spec_from_file_location("beam5_example", path)
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


def show_ledger(path):
    """Print every ledger row, sorted by overall error (best first)."""
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
        ("modes", 5, ">"),
        ("iters", 6, ">"),
        ("secs", 7, ">"),
        ("seed", 6, ">"),
        ("stop", 10, "<"),
    ]
    header = "  ".join(f"{title:{align}{width}}" for title, width, align in columns)
    print(header)
    print("-" * len(header))
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
        energy = res.get("final_energy")
        energy_str = f"{energy:12.4e}" if energy is not None else f"{'-':>12}"
        print(
            "  ".join(
                [
                    f"{_clip(cfg['name']):<{name_width}}",
                    f"{r['config_id']:>9}",
                    f"{res['overall_error']:10.3e}",
                    f"{res['worst_point_error']:10.3e}",
                    energy_str,
                    f"{res['n_modes']:5d}",
                    f"{res['total_iters']:6d}",
                    secs_str,
                    seed_str,
                    f"{res['stop_reason']:<10}",
                ]
            )
        )


def extract_result(problem, errors):
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
        }
        for r in history.stages
    ]
    return {
        "overall_error": errors["overall"],
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


def run_sweep(configs, ledger="sweep_results.jsonl", retrain=False,
              verbose=True, plot=False, checkpoint_dir=None):
    """Train every uncached config in ``configs`` and record it in ``ledger``.

    A config already present in the ledger (by ``config_id``) is skipped unless
    ``retrain`` -- the seconds-vs-minutes shortcut. Each trained config upserts
    its row and saves a checkpoint keyed by id. Returns the whole ledger.

    Args:
        configs (list[RunConfig]): the run queue.
        ledger (str or Path): JSONL file; relative paths resolve next to this
            script. A bigger, non-parameter change can target a fresh path.
        retrain (bool): train (and overwrite the row/checkpoint) even if cached.
        verbose (bool): print per-config progress and the training tables.
        plot (bool): draw the example's figures for each trained config.
        checkpoint_dir (str or Path, optional): where ``pgd5_sweep_<id>.pt``
            live; defaults to the example's ``PARAM_SWEEP_DIR``.

    Returns:
        list[dict]: every ledger row after the run.
    """
    ex = _example()
    ledger = Path(ledger)
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
        checkpoint = checkpoint_dir / f"pgd5_sweep_{cid}.pt"
        started = _time.monotonic()
        problem = ex.main(
            verbose=verbose,
            plot=plot,
            trainer_cls=trainer_cls,
            config=cfg,
            checkpoint=checkpoint,
            retrain=retrain,
        )
        errors = ex.relative_errors(problem.pgd)
        row = {
            "config_id": cid,
            "config": dataclasses.asdict(cfg),
            "result": extract_result(problem, errors),
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


def load_run(config_id, ledger="sweep_results.jsonl", checkpoint_dir=None):
    """Reload a sweep row's trained ``Problem`` from its checkpoint.

    Looks up ``config_id`` in ``ledger``, rebuilds its ``RunConfig`` from the
    stored row, and loads the matching checkpoint -- no retraining, seconds not
    minutes, same mechanism ``run_sweep`` relies on.

    Args:
        config_id (str): the row's ``config_id``, e.g. from ``show_ledger``.
        ledger (str or Path): JSONL file; relative paths resolve next to this
            script.
        checkpoint_dir (str or Path, optional): where ``pgd5_sweep_<id>.pt``
            live; defaults to the example's ``PARAM_SWEEP_DIR``, matching
            ``run_sweep``.

    Returns:
        Problem: with ``.pgd`` and ``.history`` filled from the checkpoint.
    """
    ex = _example()
    ledger = Path(ledger)
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


def show_run(config_id, ledger="sweep_results.jsonl", checkpoint_dir=None):
    """Reprint one row's full training log -- the per-stage table.

    The same output the run itself printed, regenerated from its checkpoint
    rather than from a retrain, because the ``TrainingHistory`` rides along in
    the checkpoint. Use it when the ledger's one-line summary is not enough:
    the per-stage table is where ``max corr`` lives, and a run whose modes are
    copies of each other reads 1.0 there while its energy looks perfectly fine.

    Args:
        config_id (str): the row's ``config_id``, e.g. from :func:`show_ledger`.
        ledger (str or Path): JSONL file the row lives in.
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


def plot_losses(config_id, ledger="sweep_results.jsonl", checkpoint_dir=None,
                 save_path=None):
    """Redraw the convergence curve (energy vs. iteration) for one sweep row.

    Thin wrapper around the example's ``plot_convergence``, fed by a checkpoint
    reload instead of a live training run.

    Args:
        config_id (str): the row's ``config_id``.
        ledger (str or Path): JSONL file the row lives in.
        checkpoint_dir (str or Path, optional): passed through to
            :func:`load_run`.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "pgd5_sweep_<config_id>_convergence.png"``.
    """
    ex = _example()
    kwargs = {"ledger": ledger}
    if checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = checkpoint_dir
    problem = load_run(config_id, **kwargs)
    ex.plot_convergence(
        problem.history,
        save_path=save_path or ex.PLOT_DIR / f"pgd5_sweep_{config_id}_convergence.png",
    )


def plot_extremes(config_id, ledger="sweep_results.jsonl", checkpoint_dir=None,
                    labels=None, save_path=None):
    """Redraw the two-hardest-points comparison for one sweep row.

    Thin wrapper around the example's ``plot_solution``, fed by a checkpoint
    reload instead of a live training run.

    Args:
        config_id (str): the row's ``config_id``.
        ledger (str or Path): JSONL file the row lives in.
        checkpoint_dir (str or Path, optional): passed through to
            :func:`load_run`.
        labels (list[str], optional): which reference points to draw; defaults
            to the reference bundle's highlighted pair.
        save_path (str or Path, optional): where to write the PNG; defaults to
            ``PLOT_DIR / "pgd5_sweep_<config_id>_vs_reference.png"``.

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
        save_path=save_path or ex.PLOT_DIR / f"pgd5_sweep_{config_id}_vs_reference.png",
    )


def _configs():
    """The run queue -- edit this to enter configurations.

    Removing an entry leaves its ledger row intact; changing a knob writes a new
    row beside the old one. Built lazily so the example module loads only when
    the sweep actually runs.
    """
    ex = _example()
    return [
        ex.RunConfig(
            name="simultaneous-stage_tol1e-6-enrichment_tol1e=5-seed5", strategy="simultaneous",
            min_iter=300, max_iter = 700, stage_tol=1e-6, enrichment_tol=1e-5, seed_amplitude=5.0,
        ),
        # ex.RunConfig(
        #     name="simultaneous-tol1e-5-seed0.05", strategy="simultaneous",
        #     min_iter=300, stage_tol=1e-5, enrichment_tol=1e-5, seed_amplitude=0.05,
        # ),
        # ex.RunConfig(
        #     name="simultaneous-tol1e-5-seed0.05", strategy="simultaneous",
        #     min_iter=300, stage_tol=1e-5, enrichment_tol=1e-5, seed_amplitude=0.05,
        # ),
    ]


if __name__ == "__main__":
    import sys

    retrain = "--retrain" in sys.argv[1:]
    run_sweep(_configs(), retrain=retrain)
    show_ledger(HERE / "sweep_results.jsonl")
