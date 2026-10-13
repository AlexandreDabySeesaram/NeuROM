"""V1: does NeuROM solve the beam on a foundation under the moving load?

Plan: `Plan-tests-barriere-kolmogorov.md` (notes), step V1. For k = 1e4,
a = 0.5 and P = 0.5 P_cr then 0.9 P_cr, solves the discrete problem of
`build_problem` with `minimize` (LBFGS alone) from w = 0, compares w and Pi
with the semi-analytical `reference_solution`, and measures the cost of one
solve. Same solver as `snapshots.py`.

LBFGS is preconditioned by the change of variable q = T z, T = L^-T with
K_0 = L L^T the linear stiffness: the Hessian seen by LBFGS, T^T H T, has a
condition number ~2 to 5 instead of ~5e7 at 100 elements, hence ~15 evaluations
of Pi instead of ~1e4 (see `Plan-tests-preconditionnement.md`).
FourPoints1D integrates every term exactly (k int w^2 and int f w are of
degree 6 per element).

Usage:
    uv run python check_reference.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import minimize, grad_hessian, sample_deflection, cholesky_preconditioner, precondition, unprecondition
from moving_load import bspline_load, linear_stiffness, critical_load, load_scale
from reference import reference_solution


K = 1e4
S_LOAD = 0.02         # B-spline width (= 2 h): its breakpoints are nodes
A_LOAD = 0.5          # a node
P_RATIOS = [0.5, 0.9]
N_ELEMENTS = 100
N_PER_ELEMENT = 12     # sample_deflection grid = reference grid (N_ELEMENTS * N_PER_ELEMENT steps)
TOL_CHECK = 1e-4       # expected relative error on w and Pi

# LBFGS stops on |dPi| / |Pi| <= 1e-12 (without T: reached at ~1e-5 error on w,
# just before torch's LBFGS stalls; with T: at the discretization error)
LBFGS_ITERS = 5000
LBFGS_TOL = 1e-12


def solve(P, load, T):
    """Preconditioned LBFGS solve from w = 0; returns the fine sampling of w, Pi, diagnostics and timings."""
    t0 = time.perf_counter()
    problem = build_problem(N_ELEMENTS, P, k=K, load=load, quadrature=FourPoints1D())
    t1 = time.perf_counter()
    precondition(problem["u"], T)
    history = minimize(problem["model"], adam_lr=0.0, adam_iters=0, lbfgs_lr=1.0,
                       lbfgs_iters=LBFGS_ITERS, tol=LBFGS_TOL, verbose=False)
    unprecondition(problem["u"])
    t2 = time.perf_counter()

    problem["domain"].interpolate_all(problem["field_layout"])
    _, g, H = grad_hessian(problem["model"])
    lambda_min = torch.linalg.eigvalsh(0.5 * (H + H.T))[0].item()
    x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
    return {
        "x": x, "w": w, "energy": problem["model"]().item(), "grad_norm": g.norm().item(),
        "lambda_min": lambda_min, "iters": len(history["loss"]), "converged": history["converged"],
        "t_build": t1 - t0, "t_solve": t2 - t1,
    }


def main():
    P_cr, _ = critical_load(K, N_ELEMENTS)
    F = load_scale(K, S_LOAD, N_ELEMENTS)
    load = bspline_load(F, S_LOAD, A_LOAD)
    T = cholesky_preconditioner(linear_stiffness(K, N_ELEMENTS))
    print(f"k = {K:.0e}, a = {A_LOAD}, s = {S_LOAD}, n_el = {N_ELEMENTS}, P_cr = {P_cr:.4f}, F = {F:.4f}\n")

    for ratio in P_RATIOS:
        P = ratio * P_cr
        ref = reference_solution(K, P, load, n_steps=N_ELEMENTS * N_PER_ELEMENT)
        out = solve(P, load, T)
        assert torch.allclose(out["x"], ref["x"])

        err_w = ((out["w"] - ref["w"]).norm() / ref["w"].norm()).item()
        err_max = abs(out["w"].abs().max() - ref["w"].abs().max()).item() / ref["w"].abs().max().item()
        err_pi = abs(out["energy"] - ref["energy"]) / abs(ref["energy"])
        ok = max(err_w, err_pi) <= TOL_CHECK

        print(f"P = {ratio} P_cr = {P:.4f}")
        print(f"  reference : Pi = {ref['energy']:.10e}, max|w| = {ref['w'].abs().max():.6f}, N = {ref['N']:.6f}")
        print(f"  NeuROM    : Pi = {out['energy']:.10e}, max|w| = {out['w'].abs().max():.6f}")
        print(f"  rel. error: w (L2) = {err_w:.2e}, max|w| = {err_max:.2e}, Pi = {err_pi:.2e}  -> "
              f"{'OK' if ok else 'ABOVE ' + str(TOL_CHECK)}")
        print(f"  stop      : LBFGS {out['iters']} it., converged = {out['converged']}, "
              f"||grad Pi|| = {out['grad_norm']:.2e}, lambda_min = {out['lambda_min']:.3e} (> 0: minimum)")
        print(f"  time      : build {out['t_build']:.2f} s, LBFGS {out['t_solve']:.2f} s\n")


if __name__ == "__main__":
    main()
