"""Plan-tests-preconditionnement: LBFGS with q = T z, variants A-D, 3 cases, 100 elements."""

import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # examples/1d_beam_hermite
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.utils import parametrize

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import minimize, grad_hessian, sample_deflection, newton
from moving_load import bspline_load, critical_load, load_scale
from reference import reference_solution

K, S, A_LOAD, N_EL, NPE = 1e4, 0.02, 0.5, 100, 12
OUT = Path(__file__).resolve().parents[1] / "results"  # git-ignored


class Linear(torch.nn.Module):
    def __init__(self, T):
        super().__init__()
        self.register_buffer("T", T)

    def forward(self, z):
        return (self.T @ z.flatten()).view_as(z)

    def right_inverse(self, q):
        return torch.linalg.solve(self.T, q.flatten()).view_as(q)


def build(P, load):
    return build_problem(N_EL, P, k=K, load=load, quadrature=FourPoints1D())


def set_q(problem, q):
    (p,) = problem["model"].parameters()
    with torch.no_grad():
        p.copy_(q.view_as(p))


def preconditioners():
    _, _, K0 = grad_hessian(build_problem(N_EL, 0.0, k=K, quadrature=FourPoints1D())["model"])
    K0 = 0.5 * (K0 + K0.T)
    n = K0.shape[0]
    h = 1.0 / N_EL
    B = torch.ones(n); B[1::2] = 1.0 / h          # free DOFs interleaved (w, w')
    L = torch.linalg.cholesky(K0)
    return {
        "A": torch.eye(n),
        "B": torch.diag(B),
        "C": torch.diag(K0.diagonal() ** -0.5),
        "D": torch.linalg.inv(L).T,
    }


def solve(P, load, T, q_start, w_ref):
    pb = build(P, load)
    set_q(pb, q_start)
    parametrize.register_parametrization(pb["u"], "values_reduced", Linear(T))
    model = pb["model"]
    n_eval = [0]
    model.register_forward_hook(lambda *args: n_eval.__setitem__(0, n_eval[0] + 1))
    curve = []

    def err_w():
        _, w = sample_deflection(pb, n_per_element=NPE)
        return ((w - w_ref).norm() / w_ref.norm()).item()

    def cb(phase, i, loss):
        if i % 5 == 0:
            curve.append((n_eval[0], err_w()))

    t0 = time.perf_counter()
    hist = minimize(model, adam_lr=0.0, adam_iters=0, lbfgs_lr=1.0, lbfgs_iters=5000,
                    tol=1e-12, verbose=False, callback=cb)
    t = time.perf_counter() - t0
    evals = n_eval[0]
    _, gz, Hz = grad_hessian(model)
    gq = torch.linalg.solve(T.T, gz)
    ev = torch.linalg.eigvalsh(0.5 * (Hz + Hz.T))
    e = err_w()
    curve.append((evals, e))
    return {"steps": len(hist["loss"]), "evals": evals, "time": t, "err_w": e,
            "energy": hist["loss"][-1], "grad_q": gq.norm().item(),
            "cond": (ev[-1] / ev[0]).item(), "converged": hist["converged"], "curve": curve}


def main():
    P_cr, _ = critical_load(K, N_EL)
    F = load_scale(K, S, N_EL)
    load = bspline_load(F, S, A_LOAD)
    Ts = preconditioners()
    n = Ts["A"].shape[0]
    print(f"P_cr = {P_cr:.4f}, F = {F:.3f}, n = {n}")

    # Reference for case 3: newton at 0.9 P_cr from 0, then newton at 1.2 P_cr from it
    pb9 = build(0.9 * P_cr, load); newton(pb9["model"], tol=1e-10)
    (p9,) = pb9["model"].parameters(); q09 = p9.detach().flatten().clone()
    pb12 = build(1.2 * P_cr, load); set_q(pb12, q09); it, ok = newton(pb12["model"], tol=1e-10)
    _, _, H12 = grad_hessian(pb12["model"])
    _, w12 = sample_deflection(pb12, n_per_element=NPE)
    print(f"newton 1.2 P_cr: {it} it., converged {ok}, lambda_min {torch.linalg.eigvalsh(H12)[0]:.3f}, "
          f"Pi = {pb12['model']().item():.10e}")

    cases = []
    for name, ratio, q0 in [("1", 0.5, torch.zeros(n)), ("2", 0.9, torch.zeros(n)), ("3", 1.2, q09)]:
        P = ratio * P_cr
        if ratio < 1:
            ref = reference_solution(K, P, load, n_steps=N_EL * NPE)
            w_ref, pi_ref = ref["w"], ref["energy"]
        else:
            w_ref, pi_ref = w12, pb12["model"]().item()
        cases.append((name, ratio, P, q0, w_ref, pi_ref))

    results = {}
    for name, ratio, P, q0, w_ref, pi_ref in cases:
        print(f"\n=== case {name}: P = {ratio} P_cr")
        print(f"  {'':2} {'steps':>6} {'evals':>7} {'time s':>7} {'err w':>9} {'err Pi':>9} "
              f"{'|grad_q|':>9} {'cond(THT)':>10} conv")
        for v, T in Ts.items():
            r = solve(P, load, T, q0, w_ref)
            r["err_pi"] = abs(r["energy"] - pi_ref) / abs(pi_ref)
            results[(name, v)] = r
            print(f"  {v:2} {r['steps']:6d} {r['evals']:7d} {r['time']:7.1f} {r['err_w']:9.2e} "
                  f"{r['err_pi']:9.2e} {r['grad_q']:9.2e} {r['cond']:10.2e} {r['converged']}", flush=True)
    with open(OUT / "precond_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
