"""Step 1: V1 at 100 elements, B-spline load, LBFGS alone (minimize, adam_iters=0).

Runs LBFGS with tol = 0 for a fixed budget and records, along the iterations,
the relative loss change, ||grad Pi|| and the error on w against the reference,
so the stopping criterion can be judged a posteriori.
"""

import sys
import time
from pathlib import Path

ROOT = Path("/Users/solal/Desktop/neurom/nl_pgd_beam_buckling/examples/1d_beam_hermite")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kolmogorov_study"))

import torch

from neurom.quadratures import ThreePoints1D, FourPoints1D

from main import build_problem
from utils import minimize, grad_hessian, sample_deflection
from moving_load import critical_load
from reference import reference_solution


K, S_LOAD, A_LOAD, N_EL, NPE = 1e4, 0.02, 0.5, 100, 12
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
RECORD_EVERY = 10


def bspline_load(F, s, a):
    def f(x):
        t = torch.abs((x - a) / s)
        return F * (0.5 * torch.clamp(1.5 - t, min=0) ** 2 - 1.5 * torch.clamp(0.5 - t, min=0) ** 2) / s
    return f


def load_scale(k, s, quad):
    problem = build_problem(N_EL, 0.0, k=k, load=bspline_load(1.0, s, 0.5), quadrature=quad)
    _, g, Kmat = grad_hessian(problem["model"])
    w = -torch.linalg.solve(Kmat, g)
    return 1.0 / w[0::2].abs().max().item()


def newton_exact(P, load, quad):
    """Discrete minimizer by Newton (diagnostic only: separates optimizer and discretization errors)."""
    from utils import newton
    pb = build_problem(N_EL, P, k=K, load=load, quadrature=quad)
    newton(pb["model"], tol=1e-10)
    x, w = sample_deflection(pb, n_per_element=NPE)
    return w, pb["model"]().item()


def run(P, load, ref, quad):
    pb = build_problem(N_EL, P, k=K, load=load, quadrature=quad)
    model = pb["model"]
    _, g0, _ = grad_hessian(model, hessian=False)
    rec = []

    def cb(phase, i, loss):
        if i % RECORD_EVERY == 0 or i == BUDGET - 1:
            _, g, _ = grad_hessian(model, hessian=False)
            _, w = sample_deflection(pb, n_per_element=NPE)
            err = ((w - ref["w"]).norm() / ref["w"].norm()).item()
            rec.append((i, loss, g.norm().item(), err))

    t0 = time.perf_counter()
    hist = minimize(model, adam_lr=0.0, adam_iters=0, lbfgs_lr=1.0, lbfgs_iters=BUDGET,
                    tol=0.0, verbose=False, callback=cb)
    t = time.perf_counter() - t0
    L = hist["loss"]
    rel = [abs(L[i] - L[i - 1]) / abs(L[i]) for i in range(1, len(L))]
    return rec, rel, t, g0.norm().item(), len(L)


def main():
    for qname, quad in [("4pts", FourPoints1D()), ("3pts", ThreePoints1D())]:
        P_cr, _ = critical_load(K, N_EL, quadrature=quad)
        F = load_scale(K, S_LOAD, quad)
        load = bspline_load(F, S_LOAD, A_LOAD)
        print(f"=== {qname}: P_cr = {P_cr:.6f}, F = {F:.4f}")
        for ratio in [0.5, 0.9]:
            P = ratio * P_cr
            ref = reference_solution(K, P, load, n_steps=N_EL * NPE)
            ref4 = reference_solution(K, P, load, n_steps=4 * N_EL * NPE)
            ref_self = ((ref["w"] - ref4["w"][::4]).norm() / ref4["w"].norm()).item()
            w_n, pi_n = newton_exact(P, load, quad)
            err_disc = ((w_n - ref["w"]).norm() / ref["w"].norm()).item()
            err_disc_pi = abs(pi_n - ref["energy"]) / abs(ref["energy"])
            print(f"P = {ratio} P_cr: ref self-err (1200 vs 4800 steps) {ref_self:.1e}, "
                  f"Newton discrete minimizer: err w {err_disc:.2e}, err Pi {err_disc_pi:.2e}")
            if qname == "3pts":
                continue
            rec, rel, t, g0, n = run(P, load, ref, quad)
            print(f"  LBFGS {n} steps in {t:.1f} s, ||grad Pi_0|| = {g0:.2e}")
            print(f"  {'it':>6} {'Pi':>18} {'||grad||':>10} {'err w':>10}")
            for i, loss, gn, err in rec[:: max(1, len(rec) // 25)] + [rec[-1]]:
                print(f"  {i:6d} {loss:18.10e} {gn:10.2e} {err:10.2e}")
            for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
                stop = next((i + 1 for i, r in enumerate(rel) if r <= tol), None)
                if stop is None:
                    print(f"  tol {tol:.0e}: never reached")
                    continue
                near = min(rec, key=lambda r: abs(r[0] - stop))
                print(f"  tol {tol:.0e}: would stop at it {stop}; at it {near[0]}: "
                      f"||grad|| {near[2]:.2e}, err w {near[3]:.2e}")
            print(f"  final err Pi = {abs(rec[-1][1] - ref['energy']) / abs(ref['energy']):.2e}")


if __name__ == "__main__":
    main()
