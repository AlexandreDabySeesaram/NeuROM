import importlib.util, pathlib

p = pathlib.Path("/Users/solal/Desktop/phd/projects/neurom/docs/examples/1d_5-parametric_beam_PGD/sweep.py")
sw = importlib.util.spec_from_file_location("sw", p)
mod = importlib.util.module_from_spec(sw); sw.loader.exec_module(mod)

LEDGER = "/Users/solal/Desktop/phd/projects/neurom/docs/examples/1d_5-parametric_beam_PGD/sweep_results.jsonl"

# for cid in ("62f0c902", "462540f0", "1918a2ed", "b70a0964"):
#     removed = mod.delete_row(LEDGER, cid)
#     print(f"{'deleted' if removed else 'absent'} : {cid}")

mod.show_ledger(LEDGER)   # the table

# Plot one row's convergence curve or its two-hardest-points comparison,
# reloaded from its checkpoint (no retraining):
# mod.plot_losses("462540f0", ledger=LEDGER)
# mod.plot_extremes("462540f0", ledger=LEDGER)