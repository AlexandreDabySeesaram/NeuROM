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

import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _example():
    """Load the sibling example module by path (no package install needed)."""
    path = HERE / "1d_5-parametric_beam_deflection_PGD.py"
    spec = importlib.util.spec_from_file_location("beam5_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
