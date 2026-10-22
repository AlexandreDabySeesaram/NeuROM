"""Bifurcation diagram w(1/4) - mu at P = 2200 with a slight transverse load (imperfection).

Curiosity, no experiment note. The perfect beam (`branches_k.py`) has a pitchfork
at every jump of the global minimum and an arbitrary side after it; a small
load F b_s(x - A_LOAD) (`bspline_load`, off-centre so that it acts on both the
symmetric and the antisymmetric modes) should fix the side and unfold the
pitchforks.

F: linear deflection max |w| = W_IMPERFECTION at k = 0, P = 0 (`load_scale`,
load at mid-span), against max |w| ~ 30 for the buckled beam at P = 2200.

mu = k^(1/4) on [0, MU_FACTOR mu_cr] (with the load, w != 0 beyond mu_cr), swept
both ways: increasing from the first mu, then decreasing from the last solution
of the increasing sweep. Every solve as variant (c) of `Exp-winkler-continuation`:
q_0 = q(previous mu) + EPS xi, a second solve from q + RETRY_EPS xi when
lambda_min <= 0. A difference between the two sweeps = hysteresis at the jumps.

Writes `results/fondation/imperfection_P{P}_a{A_LOAD}_w{W_IMPERFECTION}.csv` (one row per
direction and mu) and `.pt` (x, mu and the deflections W per direction).

Usage:
    uv run python imperfection_fondation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import cholesky_preconditioner, sample_deflection
from moving_load import bspline_load, linear_stiffness, load_scale
from snapshots import solve
from branches_k import critical_stiffness, PROBE


P = 2200.0
N_MU = 150
MU_FACTOR = 1.1        # mu up to 1.1 mu_cr
W_IMPERFECTION = 10.0  # linear deflection of the load at k = 0, P = 0 (earlier runs: 1e-2, 1e-1, 1)
A_LOAD = 0.3           # load position, a node (h = 0.01) (first run: 0.25)
S_LOAD = 0.02          # B-spline width (= 2 h), as in E1
EPS = 1e-4
RETRY_EPS = 1e-2
SEED = 0
N_ELEMENTS = 100
N_PER_ELEMENT = 12     # sampling of the deflections (1201 points)

RESULTS = Path(__file__).parent / "results" / "fondation"


def sweep(mu_values, q_start, load, xi, direction):
    """Continuation along mu_values from q_start (None: EPS xi). Returns the rows, the deflections and the last q."""
    rows, W, q_prev = [], [], q_start
    for mu in mu_values:
        k = mu.item() ** 4
        problem = build_problem(N_ELEMENTS, P, k=k, load=load, quadrature=FourPoints1D())
        T = cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS))
        q0 = EPS * xi if q_prev is None else q_prev + EPS * xi
        q, out = solve(problem, q0, T, stability=True)
        retried = out["lambda_min"] <= 0
        if retried:
            q, out = solve(problem, q + RETRY_EPS * xi, T, stability=True)
        rows.append({"direction": direction, "P": P, "mu": mu.item(), "k": k, "retried": retried,
                     "w_quarter": q[PROBE].item(), **out})
        x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
        W.append(w)
        q_prev = q
    return rows, x, torch.stack(W, dim=1), q_prev


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    F = load_scale(0.0, S_LOAD, N_ELEMENTS, w_target=W_IMPERFECTION)
    load = bspline_load(F, S_LOAD, A_LOAD)
    n_free = linear_stiffness(0.0, N_ELEMENTS).shape[0]
    xi = torch.randn(n_free, dtype=torch.float64, generator=torch.Generator().manual_seed(SEED))
    mu_cr = critical_stiffness(P) ** 0.25
    mu_values = torch.linspace(0.0, MU_FACTOR * mu_cr, N_MU, dtype=torch.float64)

    t0 = time.perf_counter()
    up, x, W_up, q_last = sweep(mu_values, None, load, xi, "increasing")
    down, _, W_down, _ = sweep(mu_values.flip(0), q_last, load, xi, "decreasing")
    df = pd.DataFrame(up + down)
    df["F"], df["a"], df["w_imperfection"] = F, A_LOAD, W_IMPERFECTION
    name = f"imperfection_P{P:g}_a{A_LOAD:g}_w{W_IMPERFECTION:g}"
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    torch.save({"P": P, "F": F, "a": A_LOAD, "x": x, "mu": mu_values,
                "W": {"increasing": W_up, "decreasing": W_down.flip(1)}}, RESULTS / f"{name}.pt")   # both on increasing mu
    for direction, g in df.groupby("direction"):
        print(f"{direction}: retried {int(g.retried.sum())}, saddles {int((g.lambda_min <= 0).sum())}, "
              f"LBFGS it. max {g.iters.max()}, not converged {int((~g.converged).sum())}")
    print(f"P = {P:g}, F = {F:.3e}, mu_cr = {mu_cr:.3f}, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
