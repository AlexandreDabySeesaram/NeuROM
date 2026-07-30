import importlib.util, pathlib

p = pathlib.Path("/Users/solal/Desktop/phd/projects/neurom/docs/examples/1d_5-parametric_beam/PGD/sweep.py")
sw = importlib.util.spec_from_file_location("sw", p)
mod = importlib.util.module_from_spec(sw); sw.loader.exec_module(mod)

LEDGER = "/Users/solal/Desktop/phd/projects/neurom/docs/examples/1d_5-parametric_beam/PGD/sweep_results.jsonl"

## CLEAN UP
# for cid in ("c4fda8cc", ):
#     removed = mod.delete_row(LEDGER, cid)
#     print(f"{'deleted' if removed else 'absent'} : {cid}")

## SHOW TABLE
# mod.show_ledger(LEDGER)   # the table

## LOG
# One row's full per-stage table, reloaded from its checkpoint. This is where
# `max corr` lives; the ledger table above only carries the one-line summary.
for cid in ("8c42a437", ):
    mod.show_run(cid, ledger=LEDGER)

## PLOT
# Plot one row's convergence curve or its two-hardest-points comparison,
# reloaded from its checkpoint (no retraining):
# mod.plot_losses("462540f0", ledger=LEDGER)
# mod.plot_extremes("462540f0", ledger=LEDGER)

# ### Losses
# for cid in ("dfa7c7c2",):#"0933bcbc", "c4fda8cc","844cc269","34916005",
#     mod.plot_losses(cid, ledger=LEDGER)
#     # mod.plot_extremes(cid, ledger=LEDGER)
