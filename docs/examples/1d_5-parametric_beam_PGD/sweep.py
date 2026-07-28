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


def show_ledger(path):
    """Print every ledger row, sorted by overall error (best first)."""
    rows = load_ledger(path)
    rows.sort(key=lambda r: r["result"]["overall_error"])
    print(
        f"{'name':>24} {'config_id':>10} {'overall':>10} {'worst':>10} "
        f"{'modes':>6} {'iters':>7} {'stop':>10}"
    )
    for r in rows:
        res, cfg = r["result"], r["config"]
        print(
            f"{cfg['name']:>24} {r['config_id']:>10} "
            f"{res['overall_error']:10.3e} {res['worst_point_error']:10.3e} "
            f"{res['n_modes']:6d} {res['total_iters']:7d} {res['stop_reason']:>10}"
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
            live; defaults next to this script.

    Returns:
        list[dict]: every ledger row after the run.
    """
    ex = _example()
    ledger = Path(ledger)
    if not ledger.is_absolute():
        ledger = HERE / ledger
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else HERE

    known = {r["config_id"] for r in load_ledger(ledger)}
    for cfg in configs:
        cid = ex.config_id(cfg)
        if cid in known and not retrain:
            if verbose:
                print(f"cached  : {cfg.name} ({cid})")
            continue

        trainer_cls = ex.STRATEGIES[cfg.strategy]
        checkpoint = checkpoint_dir / f"pgd5_sweep_{cid}.pt"
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
                "checkpoint": checkpoint.name,
                "git_commit": _git_commit(),
                "ledger_schema": 1,
            },
        }
        upsert_row(ledger, row)
        known.add(cid)
        if verbose:
            print(f"recorded: {cfg.name} ({cid}) "
                  f"overall={row['result']['overall_error']:.3e}")

    return load_ledger(ledger)


def _configs():
    """The run queue -- edit this to enter configurations.

    Removing an entry leaves its ledger row intact; changing a knob writes a new
    row beside the old one. Built lazily so the example module loads only when
    the sweep actually runs.
    """
    ex = _example()
    return [
        ex.RunConfig(name="baseline-greedy"),
        ex.RunConfig(
            name="baseline-simultaneous", strategy="simultaneous", min_iter=300
        ),
    ]


if __name__ == "__main__":
    import sys

    retrain = "--retrain" in sys.argv[1:]
    run_sweep(_configs(), retrain=retrain)
    show_ledger(HERE / "sweep_results.jsonl")
