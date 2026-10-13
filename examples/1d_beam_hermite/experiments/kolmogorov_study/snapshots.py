"""E1: reference solutions w(x, a) on the (k, P) grid, for the POD.

Plan: `Plan-tests-barriere-kolmogorov.md` (notes), step E1. For each
foundation stiffness k and load level P / P_cr(k), solves the discrete problem
of `build_problem` with `minimize` (LBFGS alone, preconditioned by the
Cholesky factor of the linear stiffness K_0: q = T z, T = L^-T, built once per
k) for the positions a of the B-spline load, the mesh nodes in [0.05, 0.95]:
- below P_cr (unique solution): a swept in order, each solve starting from the
  solution at the previous a (the first one from w = 0);
- above P_cr (natural branch): for each a, continuation in P from the
  solution at 0.9 P_cr, each step starting from the previous one; lambda_min
  (smallest Hessian eigenvalue, > 0 on a minimum) and the jump between steps
  are recorded to detect a loss of stability.

Writes `results/runs.csv` (one row per solve) and
`results/snapshots/k{k}_P{ratio}.pt` (DOFs Q, fine sampling W, x, a, P_cr, F).

Usage:
    uv run python snapshots.py                  # full grid
    uv run python snapshots.py --k 1e4 --n-a 5  # timing test (5 of the nodes)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import minimize, grad_hessian, sample_deflection, cholesky_preconditioner, precondition, unprecondition
from moving_load import bspline_load, linear_stiffness, critical_load, load_scale


K_VALUES = [0.0, 1e2, 1e3, 1e4]
RATIOS_BELOW = [0.0, 0.5, 0.9]
RATIOS_ABOVE = [round(r, 2) for r in np.arange(0.95, 1.5001, 0.05)]  # continuation from 0.9
A_RANGE = (0.05, 0.95) # a = the nodes in this range (91 positions)
S_LOAD = 0.02          # B-spline width (= 2 h): its breakpoints are nodes when a is a node
N_ELEMENTS = 100
N_PER_ELEMENT = 12     # fine sampling of w for the POD (1201 points)
LBFGS_ITERS = 5000
LBFGS_TOL = 1e-12      # |dPi| / |Pi|; with T, reached at the discretization error (see V1)

RESULTS = Path(__file__).parent / "results"


def solve(problem, q_start, T, stability=False):
    """LBFGS solve of `problem` from the free DOFs q_start, in place, with q = T z.

    The diagnostics (||grad Pi||, lambda_min) are computed on q, after the
    change of variable is removed.

    Returns:
        tuple: solution q (flat), dict of diagnostics.
    """
    model = problem["model"]
    (p,) = model.parameters()
    with torch.no_grad():
        p.copy_(q_start.view_as(p))

    t0 = time.perf_counter()
    precondition(problem["u"], T)
    history = minimize(model, adam_lr=0.0, adam_iters=0, lbfgs_lr=1.0,
                       lbfgs_iters=LBFGS_ITERS, tol=LBFGS_TOL, verbose=False)
    unprecondition(problem["u"])
    t_solve = time.perf_counter() - t0

    (p,) = model.parameters()
    problem["domain"].interpolate_all(problem["field_layout"])
    q = p.detach().flatten().clone()
    _, g, H = grad_hessian(model, hessian=stability)
    out = {
        "iters": len(history["loss"]), "converged": history["converged"], "grad_norm": g.norm().item(),
        "energy": model().item(), "max_w": problem["u"].full_values()[0::2].abs().max().item(),
        "t_solve": t_solve,
    }
    if stability:
        out["lambda_min"] = torch.linalg.eigvalsh(0.5 * (H + H.T))[0].item()
    return q, out


def save_snapshot(k, ratio, P, P_cr, F, a_values, Q, W, x):
    """Write the snapshots of one (k, P / P_cr)."""
    path = RESULTS / "snapshots" / f"k{k:.0e}_P{ratio:.2f}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"k": k, "ratio": ratio, "P": P, "P_cr": P_cr, "F": F, "s": S_LOAD,
                "n_elements": N_ELEMENTS, "a": a_values, "x": x, "Q": Q, "W": W}, path)


def run_k(k, n_a):
    """All load levels of one k. Returns the rows of runs.csv."""
    P_cr, _ = critical_load(k, N_ELEMENTS)
    F = load_scale(k, S_LOAD, N_ELEMENTS)
    T = cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS))
    x_nodes = torch.linspace(0.0, 1.0, N_ELEMENTS + 1)
    a_values = x_nodes[round(A_RANGE[0] * N_ELEMENTS):round(A_RANGE[1] * N_ELEMENTS) + 1]
    if n_a is not None:
        a_values = a_values[torch.linspace(0, len(a_values) - 1, n_a).round().long()]
    print(f"k = {k:.0e}: P_cr = {P_cr:.4f}, F = {F:.4f}", flush=True)
    rows = []

    def problem_at(P, a):
        return build_problem(N_ELEMENTS, P, k=k, load=bspline_load(F, S_LOAD, a.item()), quadrature=FourPoints1D())

    def record(ratio, P, a, out, **extra):
        rows.append({"k": k, "ratio": ratio, "P": P, "a": a.item(), **out, **extra})

    # Below P_cr: unique solution, warm start from the previous a
    Q_at_09 = None
    for ratio in RATIOS_BELOW:
        P = ratio * P_cr
        t0 = time.perf_counter()
        Q, W = [], []
        q = None
        for a in a_values:
            problem = problem_at(P, a)
            (p,) = problem["model"].parameters()
            q, out = solve(problem, torch.zeros(p.numel()) if q is None else q, T)
            Q.append(q)
            x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
            W.append(w)
            record(ratio, P, a, out)
        Q, W = torch.stack(Q, dim=1), torch.stack(W, dim=1)
        save_snapshot(k, ratio, P, P_cr, F, a_values, Q, W, x)
        if ratio == 0.9:
            Q_at_09 = Q
        _summary(k, ratio, rows, time.perf_counter() - t0)

    # Above P_cr: natural branch, continuation in P at fixed a
    Q_steps = {ratio: [] for ratio in RATIOS_ABOVE}
    W_steps = {ratio: [] for ratio in RATIOS_ABOVE}
    t0 = time.perf_counter()
    for j, a in enumerate(a_values):
        q = Q_at_09[:, j]
        for ratio in RATIOS_ABOVE:
            P = ratio * P_cr
            problem = problem_at(P, a)
            q_new, out = solve(problem, q, T, stability=True)
            jump = ((q_new - q).norm() / q.norm()).item()
            q = q_new
            Q_steps[ratio].append(q)
            x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
            W_steps[ratio].append(w)
            record(ratio, P, a, out, jump=jump)
    for ratio in RATIOS_ABOVE:
        save_snapshot(k, ratio, ratio * P_cr, P_cr, F, a_values,
                      torch.stack(Q_steps[ratio], dim=1), torch.stack(W_steps[ratio], dim=1), x)
    for ratio in RATIOS_ABOVE:
        _summary(k, ratio, rows, None)
    print(f"  continuation: {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def _summary(k, ratio, rows, elapsed):
    """One line per (k, P / P_cr): iterations, failures, stability."""
    df = pd.DataFrame([r for r in rows if r["k"] == k and r["ratio"] == ratio])
    line = (f"  P/P_cr = {ratio:.2f}: LBFGS it. mean {df.iters.mean():.1f} (max {df.iters.max()}), "
            f"not converged {int((~df.converged).sum())}, max||grad|| {df.grad_norm.max():.1e}, "
            f"max|w| in [{df.max_w.min():.3f}, {df.max_w.max():.3f}]")
    if "lambda_min" in df:
        line += f", min lambda_min {df.lambda_min.min():.2e}, max jump {df.jump.max():.2e}"
    if elapsed is not None:
        line += f", {elapsed:.0f} s"
    print(line, flush=True)


def _merge(path, new, k_values):
    """Replace the rows of the k values just computed, keep the others."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.k.isin(k_values)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, nargs="+", default=K_VALUES)
    parser.add_argument("--n-a", type=int, default=None, help="subset of the nodes (default: all)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    for k in args.k:
        rows = run_k(k, args.n_a)
        _merge(RESULTS / "runs.csv", pd.DataFrame(rows), [k])


if __name__ == "__main__":
    main()
