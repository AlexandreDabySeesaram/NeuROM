"""C1: snapshots w(x; k) of the unloaded beam on its foundation, for the POD in (x, k).

Plan: `Plan-tests-cas-difficile-non-lineaire.md` (notes), C1. For each axial
load P, the global minimum w(x; k) on mu = k^(1/4) in [0, mu_cr(P)) (beyond
mu_cr = k_cr^(1/4), where P_1(k) = P, the beam is straight and w = 0 adds
nothing to the POD), solved by NeuROM (`snapshots.solve`: LBFGS preconditioned
by the Cholesky factor of K_0(k)):
- start: continuation in k, q_0 = q(previous mu) + EPS xi (xi: one Gaussian
  noise, the same for every mu; EPS xi at the first mu), so that the side of the
  branch goes on from one mu to the next; when the solve ends on a saddle
  (lambda_min <= 0, the old branch just after a jump), a second solve from
  q + RETRY_EPS xi (`Exp-winkler-continuation.md`, variant (c); Avancement,
  2026-10-21); if a saddle remains, a third solve from q +- ESCAPE ||q|| v,
  v the eigenvector of lambda_min, keeping the stable one of lowest energy
  (variant (d), `Exp-winkler-calibration.md`, Déviations, 2026-11-02);
- checks: lambda_min > 0 and the error against the exact minimum c_1 phi_1(k)
  (`buckling_modes`), whose DOFs are also stored for an exact POD;
- n_jumps: number of changes of symmetry of phi_1(k) along mu (jumps of the
  global minimum between the symmetric and the antisymmetric branch).

Writes `results/c1/snapshots/P{P}.pt` (Q, W, Q_exact, x, mu, k, P, k_cr,
n_jumps) and `results/c1/runs.csv` (one row per solve).

Usage:
    uv run python snapshots_k.py             # every P of P_VALUES
    uv run python snapshots_k.py --P 250     # only these P; the rows of the other P in runs.csv are kept
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import cholesky_preconditioner, grad_hessian, sample_deflection
from moving_load import buckling_modes, linear_stiffness, membrane_stiffness
from pod import mass_matrix
from snapshots import solve
from branches_k import critical_stiffness, merge_csv


# one P in the middle of each interval between the loads P_1(mu_j) of the jumps: 0 to 9 jumps;
# 250 added: the P kept for the PGD in (x, k) (2 jumps, as 265)
P_VALUES = [70.0, 150.0, 250.0, 265.0, 425.0, 620.0, 860.0, 1135.0, 1450.0, 1800.0, 2200.0]
N_MU = 150
N_ELEMENTS = 100
N_PER_ELEMENT = 12     # fine sampling of w (1201 points), as in E1
EPS = 1e-4             # noise added to the previous solution (Exp-winkler-initialisation)
RETRY_EPS = 1e-2       # noise of the second solve from a saddle (Exp-winkler-continuation)
ESCAPE = 0.1           # step along the eigenvector of lambda_min, relative to ||q||, if a saddle remains
SEED = 0

RESULTS = Path(__file__).parent / "results" / "c1"


def exact_minimum(P, k, G):
    """DOFs of the global minimum c_1 phi_1(k) (sign of phi_1), and phi_1."""
    P_n, phi = buckling_modes(k, N_ELEMENTS, 1)
    phi = phi[:, 0]
    c = (2 * (P - P_n[0]) / (phi @ G @ phi)).clamp(min=0).sqrt()
    return c * phi, phi


def escape_saddle(problem, q, T):
    """Solves from q +- ESCAPE ||q|| v, v the eigenvector of lambda_min at the saddle q.

    The model must hold q (as left by `solve`). Keeps the stable solution of
    lowest energy and leaves it in the model; returns None if both stay saddles.
    """
    _, _, H = grad_hessian(problem["model"])
    v = torch.linalg.eigh(0.5 * (H + H.T))[1][:, 0]
    best = None
    for sign in (1.0, -1.0):
        q_s, out_s = solve(problem, q + sign * ESCAPE * q.norm() * v, T, stability=True)
        if out_s["lambda_min"] > 0 and (best is None or out_s["energy"] < best[1]["energy"]):
            best = (q_s, out_s)
    if best is not None:
        (p,) = problem["model"].parameters()
        with torch.no_grad():
            p.copy_(best[0].view_as(p))
        problem["domain"].interpolate_all(problem["field_layout"])
    return best


def is_symmetric(phi):
    """Symmetry of a mode under x -> 1 - x, from its deflections at the interior nodes."""
    w = phi[0::2]    # free DOFs interleaved (w, w')
    return bool((w @ w.flip(0)) > 0)


def run_P(P, G, M, noise):
    """Snapshots of one P. Returns the rows of runs.csv."""
    k_cr = critical_stiffness(P)
    mu_values = torch.linspace(0.0, k_cr ** 0.25, N_MU + 1, dtype=torch.float64)[:-1]
    rows, Q, W, Q_exact, symmetric = [], [], [], [], []
    q_prev = None
    t0 = time.perf_counter()
    for mu in mu_values:
        k = mu.item() ** 4
        T = cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS))
        problem = build_problem(N_ELEMENTS, P, k=k, quadrature=FourPoints1D())
        q0 = EPS * noise if q_prev is None else q_prev + EPS * noise
        q, out = solve(problem, q0, T, stability=True)
        retried = out["lambda_min"] <= 0
        if retried:
            q, out = solve(problem, q + RETRY_EPS * noise, T, stability=True)
        escaped = out["lambda_min"] <= 0
        if escaped:
            found = escape_saddle(problem, q, T)
            if found is not None:
                q, out = found
        q_prev = q
        q_exact, phi = exact_minimum(P, k, G)
        sign = 1.0 if q @ M @ q_exact >= 0 else -1.0
        diff = q - sign * q_exact
        err = ((diff @ M @ diff) / (q_exact @ M @ q_exact)).sqrt().item()
        x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
        Q.append(q)
        W.append(w)
        Q_exact.append(sign * q_exact)
        symmetric.append(is_symmetric(phi))
        rows.append({"P": P, "mu": mu.item(), "k": k, "sign": sign, "err_exact": err,
                     "symmetric": symmetric[-1], "retried": retried,
                     "escaped": escaped, **out})
    n_jumps = sum(a != b for a, b in zip(symmetric[:-1], symmetric[1:]))
    torch.save({"P": P, "k_cr": k_cr, "n_jumps": n_jumps, "n_elements": N_ELEMENTS,
                "mu": mu_values, "k": mu_values**4, "x": x, "Q": torch.stack(Q, dim=1),
                "W": torch.stack(W, dim=1), "Q_exact": torch.stack(Q_exact, dim=1)},
               RESULTS / "snapshots" / f"P{P:g}.pt")

    df = pd.DataFrame(rows)
    flips = int(((df.sign.diff() != 0) & (df.symmetric.diff() == 0)).iloc[1:].sum())
    print(f"P = {P:g}: k_cr = {k_cr:.4g}, {n_jumps} jumps, sign flips within a branch {flips}, "
          f"retried {int(df.retried.sum())}, escaped {int(df.escaped.sum())}, saddles {int((df.lambda_min <= 0).sum())}, "
          f"LBFGS it. max {df.iters.max()}, not converged {int((~df.converged).sum())}, "
          f"min lambda_min {df.lambda_min.min():.2e}, max err {df.err_exact.max():.1e}, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--P", type=float, nargs="+", default=P_VALUES,
                        help="axial loads to compute (default: P_VALUES); the rows of the other P are kept")
    p_values = parser.parse_args().P

    (RESULTS / "snapshots").mkdir(parents=True, exist_ok=True)
    G, M = membrane_stiffness(N_ELEMENTS), mass_matrix(N_ELEMENTS)
    n_free = G.shape[0]
    noise = torch.randn(n_free, dtype=G.dtype, generator=torch.Generator().manual_seed(SEED))

    rows = []
    for P in p_values:
        rows += run_P(P, G, M, noise)
    merge_csv(RESULTS / "runs.csv", pd.DataFrame(rows), p_values)


if __name__ == "__main__":
    main()
