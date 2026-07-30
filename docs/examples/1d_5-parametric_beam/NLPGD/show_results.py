"""Scratch viewer for the NL-PGD sweep ledger -- edit the ids below and run it.

Paths resolve relative to this file rather than being hard-coded absolutely, so
the script survives the directory being moved or checked out elsewhere.
"""

import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("nlpgd5_sweep", HERE / "sweep.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LEDGER = HERE / "sweep_results.jsonl"

## CLEAN UP
# for cid in ():
#     removed = mod.delete_row(LEDGER, cid)
#     print(f"{'deleted' if removed else 'absent'} : {cid}")

## SHOW TABLE
mod.show_ledger(LEDGER)

## LOG -- 
# one row's full per-stage table and coefficient rows, reloaded from
## its checkpoint. This is where `max corr` and `|C|` live; the ledger table
## above only carries the one-line summary.
# for cid in ():
#     mod.show_run(cid, ledger=LEDGER)

## PLOT -- one row's convergence curve, or its two-hardest-points comparison,
## reloaded from its checkpoint (no retraining). Ids come from the table above.
for cid in ("70805545", "7b9a75f5", "46485c86"):
    mod.plot_losses(cid, ledger=LEDGER)
    # mod.plot_extremes(cid, ledger=LEDGER)


