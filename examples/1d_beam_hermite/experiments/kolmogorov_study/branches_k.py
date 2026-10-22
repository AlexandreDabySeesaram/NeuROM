"""C1: branches of the unloaded beam against the foundation stiffness k, at fixed P.

Plan: `Plan-tests-cas-difficile-non-lineaire.md` (notes), C1. Without
transverse load, branch n of the discrete problem is exactly w = c_n phi_n,
with (K + k M) phi_n = P_n(k) G phi_n (`buckling_modes`) and
c_n^2 phi_n^T G phi_n = 2 (P - P_n(k)); it exists for P_n(k) < P, its energy is
Pi_n = -(P - P_n(k))^2 / 2, and only the branch of smallest P_n is stable (the
global minimum). Here:
- exact branches on a grid uniform in mu = k^(1/4), k in [0, K_MAX]: w(1/4) of
  every branch -> `results/c1/branches.csv`; k_cr(P), where P_1(k) = P ->
  `results/c1/critical.csv`;
- NeuROM points (LBFGS preconditioned, `snapshots.solve`) at a few k per P:
  starts +0.1 phi_n (n = 1, 2, 3), the lowest energy kept, and a start from a
  random noise (what NeuROM picks alone) -> `results/c1/neurom_points.csv`,
  with the branch identified and the error against c_n phi_n.

Usage:
    uv run python branches_k.py
"""

import sys
import time
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


P_VALUES = [100.0, 500.0, 2000.0, 2200.0]
K_MAX = 1.25e6        # mu_max = 33.4 > mu_cr(2200) = 32.9
N_ELEMENTS = 100
N_MODES = 30         # modes tracked (P_n < 2000 needs 13 at k = 0; margin for the matching)
N_K_REF = 600        # exact branches, uniform in mu = k^(1/4)
N_K_NEUROM = 30      # NeuROM points per P, uniform in mu on [0, mu_cr(P)]
N_STARTS = 3         # starts +0.1 phi_n, n = 1 .. N_STARTS
START_AMPLITUDE = 0.1
NOISE = 1e-2         # amplitude of the noise start (all DOFs)
X_PROBE = 0.25       # w(1/4): nonzero for symmetric and antisymmetric modes

RESULTS = Path(__file__).parent / "results" / "c1"
PROBE = 2 * (round(X_PROBE * N_ELEMENTS) - 1)   # free DOF of w at x = 1/4 (free DOFs: (w, w') at interior nodes)


def amplitudes(P, P_n, phi, G):
    """c_n of the branches w = c_n phi_n (nan where P_n >= P)."""
    g = torch.einsum("in,ij,jn->n", phi, G, phi)
    c2 = 2 * (P - P_n) / g
    return torch.where(c2 > 0, c2.clamp(min=0).sqrt(), torch.full_like(c2, float("nan")))


def track(phi_prev, phi, G):
    """Order and signs of the modes phi matching phi_prev (largest G-overlap, greedy).

    The eigenvalue order swaps where two curves P_n(k) cross; following the
    modes by continuity keeps one label per branch.
    """
    g_prev = torch.einsum("in,ij,jn->n", phi_prev, G, phi_prev)
    g = torch.einsum("in,ij,jn->n", phi, G, phi)
    overlap = (phi_prev.T @ G @ phi) / (g_prev[:, None] * g[None, :]).sqrt()
    order = torch.full((phi.shape[1],), -1, dtype=torch.long)
    free = torch.ones(phi.shape[1], dtype=torch.bool)
    for flat in overlap.abs().flatten().argsort(descending=True):
        i, j = divmod(int(flat), phi.shape[1])
        if order[i] < 0 and free[j]:
            order[i], free[j] = j, False
    signs = torch.sign(overlap[torch.arange(len(order)), order])
    return order, signs


def exact_branches(G):
    """w(1/4) and energy of every branch, for every P, on the reference grid in mu.

    Columns: `branch` labels a branch by continuity in k (its rank at k = 0),
    `n` is the rank of P_n at this k (n = 1: global minimum).
    """
    rows = []
    phi_prev = None
    for mu in torch.linspace(0.0, K_MAX ** 0.25, N_K_REF, dtype=torch.float64):
        k = mu.item() ** 4
        P_n, phi = buckling_modes(k, N_ELEMENTS, N_MODES)
        if phi_prev is None:
            labels = torch.arange(N_MODES)
        else:
            order, signs = track(phi_prev, phi, G)
            phi = phi.clone()
            labels = torch.empty(N_MODES, dtype=torch.long)
            for label, (j, sign) in enumerate(zip(order, signs)):
                phi[:, j] *= sign
                labels[j] = label
        phi_prev = phi[:, labels.argsort()]
        for P in P_VALUES:
            c = amplitudes(P, P_n, phi, G)
            for n in range(N_MODES):
                if torch.isnan(c[n]):
                    continue
                rows.append({"P": P, "k": k, "mu": mu.item(), "n": n + 1, "branch": labels[n].item() + 1,
                             "P_n": P_n[n].item(), "w_quarter": (c[n] * phi[PROBE, n]).item(),
                             "energy": -0.5 * (P - P_n[n].item()) ** 2, "global": n == 0})
    return pd.DataFrame(rows)


def mu_critical(branches, P):
    """mu where P_1(k) = P, read on the reference grid (last mu with a buckled branch)."""
    return branches[(branches.P == P) & (branches.n == 1)].mu.max()


def critical_stiffness(P, tol=1e-10, first_load=None):
    """k_cr(P): the k where P_1(k) = P (bisection in mu); the beam is unbuckled beyond it.

    Args:
        first_load (callable, optional): k -> P_1(k); defaults to `buckling_modes`.
    """
    if first_load is None:
        first_load = lambda k: buckling_modes(k, N_ELEMENTS, 1)[0].item()
    lo, hi = 0.0, K_MAX ** 0.25
    while first_load(hi**4) < P:    # widen the bracket: P_1 grows with k
        lo, hi = hi, 2 * hi
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if first_load(mid**4) < P else (lo, mid)
    return (0.5 * (lo + hi)) ** 4


def identify(q, P, P_n, phi, G, M):
    """Branch n whose phi_n carries q (largest G-norm component), and error against c_n phi_n."""
    g = torch.einsum("in,ij,jn->n", phi, G, phi)
    alpha = (phi.T @ (G @ q)) / g
    n = int((alpha.abs() * g.sqrt()).argmax())
    c = amplitudes(P, P_n, phi, G)[n]
    exact = torch.sign(alpha[n]) * c * phi[:, n]
    diff = q - exact
    err = ((diff @ M @ diff) / (exact @ M @ exact)).sqrt().item() if not torch.isnan(c) else float("nan")
    return n + 1, err


def neurom_points(branches, G, M):
    rows = []
    for P in P_VALUES:
        mu_cr = mu_critical(branches, P)
        t0 = time.perf_counter()
        for mu in torch.linspace(0.0, mu_cr, N_K_NEUROM + 1, dtype=torch.float64)[:-1]:
            k = mu.item() ** 4
            P_n, phi = buckling_modes(k, N_ELEMENTS, N_MODES)
            T = cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS))
            starts = {f"phi{n + 1}": START_AMPLITUDE * phi[:, n] for n in range(N_STARTS)}
            noisy = build_problem(N_ELEMENTS, P, k=k, noise=NOISE, seed=0, quadrature=FourPoints1D())
            starts["noise"] = next(noisy["model"].parameters()).detach().flatten().clone()
            results = []
            for name, q0 in starts.items():
                problem = build_problem(N_ELEMENTS, P, k=k, quadrature=FourPoints1D())
                q, out = solve(problem, q0, T, stability=True)
                n, err = identify(q, P, P_n, phi, G, M)
                results.append({"P": P, "k": k, "mu": mu.item(), "start": name,
                                "w_quarter": q[PROBE].item(), "branch": n, "err_branch": err, **out})
            best = min((r for r in results if r["start"] != "noise"), key=lambda r: r["energy"])
            for r in results:
                r["kept"] = r is best
            rows += results
        print(f"P = {P:g}: mu_cr = {mu_cr:.3f} (k_cr = {mu_cr ** 4:.3g}), {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    G = membrane_stiffness(N_ELEMENTS)
    M = mass_matrix(N_ELEMENTS)

    t0 = time.perf_counter()
    branches = exact_branches(G)
    branches.to_csv(RESULTS / "branches.csv", index=False)
    print(f"exact branches: {time.perf_counter() - t0:.0f} s", flush=True)

    pd.DataFrame([{"P": P, "k_cr": (k := critical_stiffness(P)), "mu_cr": k**0.25} for P in P_VALUES]
                 ).to_csv(RESULTS / "critical.csv", index=False)

    points = neurom_points(branches, G, M)
    points.to_csv(RESULTS / "neurom_points.csv", index=False)
    for P, df in points.groupby("P"):
        kept, noise = df[df.kept], df[df.start == "noise"]
        print(f"P = {P:g}: kept on branch 1 {int((kept.branch == 1).sum())}/{len(kept)}, "
              f"max err {kept.err_branch.max():.1e}, min lambda_min {kept.lambda_min.min():.2e}, "
              f"not converged {int((~df.converged).sum())}; noise start on branch 1 "
              f"{int((noise.branch == 1).sum())}/{len(noise)}")


if __name__ == "__main__":
    main()
