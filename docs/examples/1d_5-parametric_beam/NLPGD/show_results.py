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

# Follows whatever `sweep.py` is currently writing, rather than naming a file:
# the campaign ledger has moved once already, and a hard-coded name silently
# shows a stale table instead of failing. Point it at an old file explicitly to
# read one -- `HERE / "sweep_results.jsonl"`, `HERE / "sweep_results_grid.jsonl"`.
LEDGER = HERE / mod.LEDGER

## CLEAN UP
# for cid in ("9b1ed483", ):
#     removed = mod.delete_row(LEDGER, cid)
#     print(f"{'deleted' if removed else 'absent'} : {cid}")

## SHOW TABLE
mod.show_ledger(LEDGER)

## LOG -- 
# one row's full per-stage table and coefficient rows, reloaded from
## its checkpoint. This is where `max corr` and `|C|` live; the ledger table
## above only carries the one-line summary.
# for cid in ("9b1ed483",):
#     mod.show_run(cid, ledger=LEDGER)

## LEADING COEFFICIENTS -- c_m per stage, one column per mode: did they settle?
## Only says something for rows run with `leading_coefficient=True`; it prints
## why rather than printing zeros otherwise.
# for cid in ():
#     mod.show_leading_coefficients(cid, ledger=LEDGER)

## PLOT -- one row's convergence curve, or its two-hardest-points comparison,
## reloaded from its checkpoint (no retraining). Ids come from the table above.
## The ids that were here belong to `sweep_results.jsonl`, so they raise a
## KeyError against the current ledger -- refill from the table above.
# for cid in ():
#     mod.plot_losses(cid, ledger=LEDGER)
#     mod.plot_extremes(cid, ledger=LEDGER)


