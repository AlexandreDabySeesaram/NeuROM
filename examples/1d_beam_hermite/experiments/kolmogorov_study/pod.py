"""E2: POD of the snapshots w(x, a), decay of the best rank-n approximation error.

Plan: `Plan-tests-barriere-kolmogorov.md` (notes), step E2. For each
(k, P / P_cr) of E1, the best rank-n approximation
w_n(x, a) = sum_{m <= n} sigma_m phi_m(x) psi_m(a) in L2(Omega_x x Omega_a)
is given by the SVD of the snapshots written in orthonormal coordinates:
- in x, exactly: int w^2 dx = q^T M q (M: Hermite mass matrix) = ||L^T q||^2
  with M = L L^T, so the rows of L^T Q are orthonormal coordinates;
- in a, by the trapezoidal rule on the 91 positions: columns scaled by
  sqrt(trapezoid weight).
Then eps(n) = sqrt(sum_{m > n} sigma_m^2 / sum_m sigma_m^2) (Eckart-Young).

Check on every file: the plain SVD of the fine sampling W (trapezoidal rule in
x too) must give the same normalised sigma.

Writes `results/pod.csv` (one row per (k, P / P_cr): sigma_1..sigma_91, n(1e-2),
n(1e-3), eps(10)) and `results/pod_modes/k{k}_P{ratio}.pt` (first modes phi
sampled on x, psi on a).

Usage:
    uv run python pod.py             # all the snapshot files
    uv run python pod.py --witness   # rank-1 witness w(x, P) = c(P) psi_1(x) only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from modes import MODES, exact_solution
from moving_load import linear_stiffness


TOLERANCES = [1e-2, 1e-3]
N_REPORT = 10   # eps(10) in pod.csv, number of modes saved in pod_modes/

RESULTS = Path(__file__).parent / "results"


def mass_matrix(n_elements):
    """Hermite mass matrix M on the free DOFs: int w v dx = q^T M q_v.

    The foundation term k/2 int w^2 of the energy has Hessian k M, so M is the
    difference of the linear stiffnesses at k = 1 and k = 0 (exact with
    `FourPoints1D`).
    """
    return linear_stiffness(1.0, n_elements) - linear_stiffness(0.0, n_elements)


def trapezoid_weights(t):
    """Weights of the trapezoidal rule on the sorted points t: int f = sum_j weights_j f(t_j)."""
    dt = t[1:] - t[:-1]
    weights = torch.zeros_like(t)
    weights[:-1] += dt / 2
    weights[1:] += dt / 2
    return weights


def weighted_svd(A, weights_a):
    """SVD of the snapshots A (rows: orthonormal coordinates in x, columns: points in a).

    Returns:
        tuple: sigma, left singular vectors U, right functions psi (sampled on
        a, orthonormal for the trapezoidal rule), so that
        A = U diag(sigma) psi^T diag(weights_a).
    """
    root = weights_a.sqrt()
    U, sigma, Vh = torch.linalg.svd(A * root[None, :], full_matrices=False)
    return sigma, U, Vh.T / root[:, None]


def decay(sigma):
    """eps(n) for n = 0 .. len(sigma): relative L2 error of the best rank-n approximation."""
    tail = (sigma**2).flip(0).cumsum(0).flip(0)                 # sum_{m >= n+1} sigma_m^2, n = 0..
    return torch.cat([tail, tail.new_zeros(1)]).div(tail[0]).sqrt()


def rank_for(eps, tol):
    """Smallest n with eps(n) <= tol."""
    return int(torch.nonzero(eps <= tol)[0])


def pod_file(path, L, weights_x):
    """POD of one snapshot file. Returns the row of pod.csv and the saved modes."""
    data = torch.load(path, weights_only=False)
    Q, W, x, a = data["Q"], data["W"], data["x"], data["a"]
    weights_a = trapezoid_weights(a)

    sigma, _, psi = weighted_svd(L.T @ Q, weights_a)
    sigma_check, _, _ = weighted_svd(W * weights_x.sqrt()[:, None], weights_a)
    eps = decay(sigma)

    n = min(N_REPORT, len(sigma))
    phi = (W * weights_a[None, :]) @ psi[:, :n] / sigma[:n]   # phi_m = int w psi_m da / sigma_m
    row = {
        "k": data["k"], "ratio": data["ratio"], "P": data["P"], "P_cr": data["P_cr"],
        **{f"n_{tol:.0e}": rank_for(eps, tol) for tol in TOLERANCES},
        f"eps_{N_REPORT}": eps[N_REPORT].item(),
        "check_sampled": (sigma / sigma.norm() - sigma_check / sigma_check.norm()).abs().max().item(),
        **{f"sigma_{m + 1}": s.item() for m, s in enumerate(sigma)},
    }
    modes = {"k": data["k"], "ratio": data["ratio"], "x": x, "a": a, "sigma": sigma,
             "phi": phi, "psi": psi[:, :n]}
    return row, modes


def witness(n_x=1201, n_P=91):
    """Rank-1 witness: the clamped beam without foundation nor load, w(x, P) = c(P) psi_1(x).

    Sampled snapshots, trapezoidal rule in x and in P. Expected eps(1) at
    round-off.
    """
    _, gamma = MODES["psi1"]
    x = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
    P = torch.linspace(0.0, 1.5 * gamma**2, n_P, dtype=torch.float64)
    W = exact_solution(x, P)
    sigma, _, _ = weighted_svd(W * trapezoid_weights(x).sqrt()[:, None], trapezoid_weights(P))
    eps = decay(sigma)
    print(f"witness w(x, P) = c(P) psi_1(x): sigma_1 = {sigma[0]:.4e}, sigma_2 = {sigma[1]:.1e}, "
          f"eps(1) = {eps[1]:.1e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness", action="store_true", help="run the rank-1 witness only")
    args = parser.parse_args()

    witness()
    if args.witness:
        return

    paths = sorted((RESULTS / "snapshots").glob("*.pt"))
    first = torch.load(paths[0], weights_only=False)
    M = mass_matrix(first["n_elements"]).to(torch.float64)
    L = torch.linalg.cholesky(0.5 * (M + M.T))
    weights_x = trapezoid_weights(first["x"])

    rows = []
    (RESULTS / "pod_modes").mkdir(exist_ok=True)
    for path in paths:
        row, modes = pod_file(path, L, weights_x)
        torch.save(modes, RESULTS / "pod_modes" / path.name)
        rows.append(row)
        print(f"k = {row['k']:.0e}, P/P_cr = {row['ratio']:.2f}: n(1e-2) = {row['n_1e-02']:2d}, "
              f"n(1e-3) = {row['n_1e-03']:2d}, eps(10) = {row[f'eps_{N_REPORT}']:.1e}, "
              f"check W = {row['check_sampled']:.1e}", flush=True)
    pd.DataFrame(rows).sort_values(["k", "ratio"]).to_csv(RESULTS / "pod.csv", index=False)


if __name__ == "__main__":
    main()
