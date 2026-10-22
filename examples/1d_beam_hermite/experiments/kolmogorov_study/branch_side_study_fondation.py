"""Side of the branch along k, unloaded beam on its foundation: same noise or continuation.

Experiment: `Exp-winkler-continuation.md` (notes). At fixed P, for mu = k^(1/4)
on [0, mu_cr), the global minimum is +-c_1 phi_1(k); between two jumps (changes
of symmetry of phi_1) the "side" (sign) should go on from one mu to the next.
Two starts of the NeuROM solve (`snapshots.solve`, LBFGS preconditioned):
- (a) "same noise": q_0 = eps xi, the same noise xi for every mu;
- (b) "continuation": q_0 = q(previous mu) + eps xi;
- (c) "continuation + retry": as (b), and when the solve ends on a saddle
  (lambda_min <= 0, typically just after a jump) a second solve from
  q + RETRY_EPS xi (deviation of 2026-10-21).
Per mu: side change (same symmetry as the previous mu and q . M q_prev < 0),
lambda_min, error against the exact +-c_1 phi_1(k), w(1/4) (bifurcation diagram).

K_0(k) = K_b + k M (the foundation term is k/2 int w^2), so K_b, M and G are
assembled once.

Writes `results/fondation/side_P{P}.csv` (one row per variant and mu) and
`results/fondation/side_P{P}.pt` (x, mu, deflections W per variant, jumps).

Usage:
    uv run python branch_side_study_fondation.py
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
from moving_load import generalized_modes, linear_stiffness, membrane_stiffness
from pod import mass_matrix
from snapshots import solve
from branches_k import critical_stiffness, PROBE
from snapshots_k import is_symmetric


P = 500.0
N_MU = 150
EPS = 1e-4
RETRY_EPS = 1e-2
SEED = 0
N_ELEMENTS = 100
N_PER_ELEMENT = 12
VARIANTS = ["same noise", "continuation", "continuation + retry"]

RESULTS = Path(__file__).parent / "results" / "fondation"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    K_b, M, G = linear_stiffness(0.0, N_ELEMENTS), mass_matrix(N_ELEMENTS), membrane_stiffness(N_ELEMENTS)
    xi = torch.randn(G.shape[0], dtype=G.dtype, generator=torch.Generator().manual_seed(SEED))

    def first_mode(k):
        P_n, phi = generalized_modes(K_b + k * M, G, 1)
        return P_n[0].item(), phi[:, 0]

    k_cr = critical_stiffness(P, first_load=lambda k: first_mode(k)[0])
    mu_values = torch.linspace(0.0, k_cr ** 0.25, N_MU + 1, dtype=torch.float64)[:-1]

    # Exact minimum and symmetry of phi_1 at every mu (shared by the two variants)
    exact, symmetric = [], []
    for mu in mu_values:
        P_1, phi = first_mode(mu.item() ** 4)
        exact.append((2 * (P - P_1) / (phi @ G @ phi)).sqrt() * phi)
        symmetric.append(is_symmetric(phi))
    jumps = [0.5 * (mu_values[j] + mu_values[j + 1]).item()
             for j in range(N_MU - 1) if symmetric[j] != symmetric[j + 1]]

    rows, deflections = [], {}
    for variant in VARIANTS:
        t0 = time.perf_counter()
        W, q_prev = [], None
        for j, mu in enumerate(mu_values):
            k = mu.item() ** 4
            problem = build_problem(N_ELEMENTS, P, k=k, quadrature=FourPoints1D())
            q0 = EPS * xi if (variant == "same noise" or q_prev is None) else q_prev + EPS * xi
            T = cholesky_preconditioner(K_b + k * M)
            q, out = solve(problem, q0, T, stability=True)
            retried = variant == "continuation + retry" and out["lambda_min"] <= 0
            if retried:
                q, out = solve(problem, q + RETRY_EPS * xi, T, stability=True)
            side_change = (q_prev is not None and symmetric[j] == symmetric[j - 1]
                           and (q @ M @ q_prev).item() < 0)
            q_exact = exact[j] if (q @ M @ exact[j]) >= 0 else -exact[j]
            err = (((q - q_exact) @ M @ (q - q_exact)) / (q_exact @ M @ q_exact)).sqrt().item()
            x, w = sample_deflection(problem, n_per_element=N_PER_ELEMENT)
            W.append(w)
            rows.append({"variant": variant, "P": P, "mu": mu.item(), "k": k, "symmetric": symmetric[j],
                         "side_change": side_change, "retried": retried, "w_quarter": q[PROBE].item(),
                         "err_exact": err, **out})
            q_prev = q
        deflections[variant] = torch.stack(W, dim=1)
        df = pd.DataFrame([r for r in rows if r["variant"] == variant])
        print(f"{variant}: side changes within a segment {int(df.side_change.sum())}, "
              f"lambda_min <= 0 at {int((df.lambda_min <= 0).sum())} mu, retried {int(df.retried.sum())}, "
              f"max err (lambda_min > 0) {df[df.lambda_min > 0].err_exact.max():.1e}, "
              f"LBFGS it. max {df.iters.max()}, {time.perf_counter() - t0:.0f} s", flush=True)

    print(f"P = {P:g}: k_cr = {k_cr:.4g}, {len(jumps)} jumps at mu = {', '.join(f'{m:.2f}' for m in jumps)}")
    pd.DataFrame(rows).to_csv(RESULTS / f"side_P{P:g}.csv", index=False)
    torch.save({"P": P, "k_cr": k_cr, "x": x, "mu": mu_values, "jumps": jumps, "W": deflections},
               RESULTS / f"side_P{P:g}.pt")


if __name__ == "__main__":
    main()
