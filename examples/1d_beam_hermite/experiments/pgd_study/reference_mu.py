"""FE reference w_EF(x; mu) at LHS points, for the error of the PGD in (x, mu).

Protocol: `Exp-pgd-mu-1.md` (vault), paragraph « Erreur ». Latin
hypercube of N_POINTS points (x_i, mu_i) in [0, 1] x MU_BOX (N_POINTS strata
per direction, one point per stratum); at each mu_i the unloaded beam on its
foundation (k = mu_i^4) is solved by NeuROM without PGD (`main.build_problem`),
on the mesh of the PGD, with the protocol of the C1 snapshots
(`kolmogorov_study/snapshots_k.py`; Avancement, 2026-10-21):
- mu_i sorted, continuation q_0 = q(previous mu) + EPS xi (EPS xi at the first
  mu; xi one Gaussian noise for every mu), LBFGS preconditioned by the Cholesky
  factor of K_0(k) (`snapshots.solve`);
- when the solve ends on a saddle (lambda_min <= 0), a second solve from
  q + RETRY_EPS xi;
- check: error against the global minimum c_1 phi_1(k) on the same mesh.

Writes `results/reference/lhs_N{N}_seed{SEED_LHS}.pt` (x, mu in the order of
the draw, the DOFs Q of w_EF and the parameters) and
`results/reference/runs_N{N}_seed{SEED_LHS}.csv` (one row per solve).

Usage:
    uv run python reference_mu.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kolmogorov_study"))

import pandas as pd
import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import cholesky_preconditioner
from moving_load import buckling_modes, linear_stiffness, membrane_stiffness
from pod import mass_matrix
from snapshots import solve


P = 2200.0
N_ELEMENTS = 20          # mesh of the PGD (main_pgd_mu.py)
MU_BOX = (0.0, 40.0)     # parametric box of the PGD
N_POINTS = 25            # LHS: 25 strata per direction
SEED_LHS = 0
SEED = 0                 # noise xi
EPS = 1e-4               # noise added to the previous solution (Exp-winkler-initialisation)
RETRY_EPS = 1e-2         # noise of the second solve from a saddle (Exp-winkler-continuation)

RESULTS = Path(__file__).parent / "results" / "reference"


def lhs(n_points, n_dims, generator):
    """Latin hypercube in [0, 1]^n_dims: one point per stratum in every direction.

    Each direction is cut into `n_points` equal strata; a random permutation
    gives the stratum of every point and a uniform draw its position inside.

    Returns:
        torch.Tensor: points, shape (n_points, n_dims).
    """
    strata = torch.stack(
        [torch.randperm(n_points, generator=generator) for _ in range(n_dims)], dim=1
    )
    offsets = torch.rand(n_points, n_dims, generator=generator)
    return (strata + offsets) / n_points


def exact_minimum(k, G):
    """DOFs of the global minimum c_1 phi_1(k) on the mesh (0 beyond mu_cr)."""
    P_n, phi = buckling_modes(k, N_ELEMENTS, 1)
    phi = phi[:, 0]
    c = (2 * (P - P_n[0]) / (phi @ G @ phi)).clamp(min=0).sqrt()
    return c * phi


def solve_references(mu_values, noise, G, M):
    """Solve w_EF at every mu by continuation in increasing mu.

    Returns:
        tuple: DOFs Q (n_free, N), in the order of `mu_values`, and the rows of
        the runs file.
    """
    order = torch.argsort(mu_values)
    Q = torch.zeros(noise.shape[0], mu_values.shape[0])
    rows = []
    q_prev = None
    for i in order.tolist():
        mu = mu_values[i].item()
        k = mu**4
        T = cholesky_preconditioner(linear_stiffness(k, N_ELEMENTS))
        problem = build_problem(N_ELEMENTS, P, k=k, quadrature=FourPoints1D())
        q0 = EPS * noise if q_prev is None else q_prev + EPS * noise
        q, out = solve(problem, q0, T, stability=True)
        retried = out["lambda_min"] <= 0
        if retried:
            q, out = solve(problem, q + RETRY_EPS * noise, T, stability=True)
        q_prev = q
        Q[:, i] = q

        # Error against c_1 phi_1(k), up to the sign; absolute beyond mu_cr
        q_exact = exact_minimum(k, G)
        sign = 1.0 if q @ M @ q_exact >= 0 else -1.0
        diff = q - sign * q_exact
        norm_exact = (q_exact @ M @ q_exact).sqrt().item()
        err = (diff @ M @ diff).sqrt().item()
        rows.append({"index": i, "mu": mu, "k": k, "retried": retried,
                     "norm_exact": norm_exact,
                     "err_exact": err / norm_exact if norm_exact > 0 else err, **out})
    return Q, rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    u = lhs(N_POINTS, 2, torch.Generator().manual_seed(SEED_LHS))
    x_values = u[:, 0]
    mu_values = MU_BOX[0] + (MU_BOX[1] - MU_BOX[0]) * u[:, 1]

    G, M = membrane_stiffness(N_ELEMENTS), mass_matrix(N_ELEMENTS)
    noise = torch.randn(G.shape[0], dtype=G.dtype, generator=torch.Generator().manual_seed(SEED))
    Q, rows = solve_references(mu_values, noise, G, M)

    name = f"N{N_POINTS}_seed{SEED_LHS}"
    torch.save({"x": x_values, "mu": mu_values, "Q": Q, "P": P, "n_elements": N_ELEMENTS,
                "mu_box": MU_BOX, "n_points": N_POINTS, "seed_lhs": SEED_LHS, "seed": SEED,
                "eps": EPS, "retry_eps": RETRY_EPS},
               RESULTS / f"lhs_{name}.pt")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"runs_{name}.csv", index=False)

    print(f"{N_POINTS} solves, retried {int(df.retried.sum())}, "
          f"saddles {int((df.lambda_min <= 0).sum())}, "
          f"LBFGS it. max {df.iters.max()}, not converged {int((~df.converged).sum())}, "
          f"min lambda_min {df.lambda_min.min():.2e}, max err {df.err_exact.max():.1e}, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
