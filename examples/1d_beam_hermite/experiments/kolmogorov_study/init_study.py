"""Influence of the initialization on the branch reached by LBFGS (unloaded beam on its foundation).

Note: `Plans tests/Note-initialisation.md` (notes). Without transverse load and
above P_1, the only minima are +-c_1 phi_1; w = 0 and the branches n >= 2 are
saddles (see `branches_k.py`). Starts w_0 = A phi_n(k) + noise of amplitude eps
(Gaussian on all the DOFs):
- test 1: where does the minimization end (global minimum, saddle of a branch
  n, or w = 0)? Starts 0, 0.1 phi_1, 0.1 phi_2, 0.1 phi_3, eps in {0, 1e-8,
  1e-4}, one seed -> `results/init/test1.csv`;
- test 2: what sets the sign of the branch? Starts 0, 0.1 phi_2, 0.1 phi_3
  (their component on phi_1 comes from the noise only), eps in {1e-8, 1e-4},
  N_SEEDS seeds -> `results/init/test2.csv`. Expected: final sign = sign of
  alpha_1(q_0), the component of the start on phi_1 (alpha_n = phi_n^T G q /
  phi_n^T G phi_n), since dPi / dalpha_1 is proportional to alpha_1.

Usage:
    uv run python init_study.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import cholesky_preconditioner
from moving_load import buckling_modes, linear_stiffness, membrane_stiffness
from pod import mass_matrix
from snapshots import solve
from branches_k import identify


P = 500.0
K_VALUES = [0.0, 1e3, 1e4]
N_ELEMENTS = 100
N_MODES = 30
START_AMPLITUDE = 0.1
STARTS_1 = [0, 1, 2, 3]         # 0: w_0 = 0, n: w_0 = 0.1 phi_n
EPS_1 = [0.0, 1e-8, 1e-4]
STARTS_2 = [0, 2, 3]
EPS_2 = [1e-8, 1e-4]
N_SEEDS = 50
ZERO = 1e-6                     # max |w| at the nodes below which the solution is w = 0

RESULTS = Path(__file__).parent / "results" / "init"


def component(q, phi, G, n=0):
    """alpha_n(q): component of q on phi_n (the phi_n are G-orthogonal)."""
    return ((phi[:, n] @ G @ q) / (phi[:, n] @ G @ phi[:, n])).item()


def run(k, start, eps, seed, setup):
    """One minimization from w_0 = 0.1 phi_start (0: w_0 = 0) + eps * noise(seed)."""
    P_n, phi, T, G, M = setup
    q0 = torch.zeros(phi.shape[0], dtype=phi.dtype)
    if start > 0:
        q0 = q0 + START_AMPLITUDE * phi[:, start - 1]
    noise = torch.randn(phi.shape[0], dtype=phi.dtype, generator=torch.Generator().manual_seed(seed))
    q0 = q0 + eps * noise
    q, out = solve(build_problem(N_ELEMENTS, P, k=k, quadrature=FourPoints1D()), q0, T, stability=True)

    if q[0::2].abs().max() < ZERO:      # free DOFs interleaved (w, w')
        arrival, err = "w = 0", float("nan")
    else:
        rank, err = identify(q, P, P_n, phi, G, M)
        arrival = "global minimum" if rank == 1 else f"saddle n = {rank}"
    alpha_final = component(q, phi, G)
    return {"k": k, "start": f"0.1 phi{start}" if start else "0", "eps": eps, "seed": seed,
            "arrival": arrival, "err_branch": err, "alpha1_noise": eps * component(noise, phi, G),
            "alpha1_start": component(q0, phi, G), "alpha1_final": alpha_final,
            "sign_final": int(torch.sign(torch.tensor(alpha_final))) if arrival == "global minimum" else 0,
            "lambda_min": out["lambda_min"], "iters": out["iters"], "converged": out["converged"],
            "energy": out["energy"]}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    G, M = membrane_stiffness(N_ELEMENTS), mass_matrix(N_ELEMENTS)
    rows1, rows2 = [], []
    for k in K_VALUES:
        P_n, phi = buckling_modes(k, N_ELEMENTS, N_MODES)
        setup = (P_n, phi, cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS)), G, M)
        rows1 += [run(k, start, eps, 0, setup) for start in STARTS_1 for eps in EPS_1]
        rows2 += [run(k, start, eps, seed, setup)
                  for start in STARTS_2 for eps in EPS_2 for seed in range(N_SEEDS)]
        print(f"k = {k:g} done", flush=True)

    test1, test2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    test1.to_csv(RESULTS / "test1.csv", index=False)
    test2.to_csv(RESULTS / "test2.csv", index=False)

    print("\nTest 1: arrival (lambda_min)")
    test1["cell"] = test1.arrival + " (" + test1.lambda_min.map("{:+.3f}".format) + ")"
    for k, df in test1.groupby("k"):
        print(f"k = {k:g}")
        print(df.pivot(index="start", columns="eps", values="cell").to_string())

    print("\nTest 2: sign of the branch")
    reached = test2[test2.arrival == "global minimum"]
    summary = reached.groupby(["k", "start", "eps"]).apply(lambda d: pd.Series({
        "reached": f"{len(d)}/{N_SEEDS}",
        "sign +": (d.sign_final > 0).mean(),
        "sign = sign(alpha1_start)": (d.sign_final == d.alpha1_start.apply(lambda a: 1 if a > 0 else -1)).mean(),
        "alpha1_start = alpha1_noise": (d.alpha1_start - d.alpha1_noise).abs().max(),
    }), include_groups=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
