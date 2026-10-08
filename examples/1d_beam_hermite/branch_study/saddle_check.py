"""Stops near saddle points: `tol` criterion or true stationary point?

Plan: `Plan-tests-branch-selection-2.md` (notes). Replays the test F runs that
did not end on psi1 and writes `results/saddle_check.csv`:
- G3: exact discrete critical points (Newton on grad Pi = 0) from psi1, psi2
  and w = 0, to validate the Hessian diagnostic (expected Morse index 0, 1, 2);
- G0: the faulty runs replayed with the original criterion (must reproduce
  `runs.csv`), with ||grad Pi|| and the Hessian spectrum at the stop.

Usage:
    uv run python saddle_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from main import build_problem
from utils import minimize, sample_deflection
from modes import MODES, classify_branch, exact_amplitude
from branch_selection import (
    P_F, ADAM_LR, LBFGS_LR, TOL, OPTIMIZERS, QUADRATURES, IMPERFECTIONS,
)


NEWTON_TOL = 1e-9      # G3, float64 round-off floor of ||grad Pi|| is ~1e-10
NEG_EIG_TOL = 1e-10    # eigenvalue counted negative if < -NEG_EIG_TOL * max |eig|

N_CASES = {"tol": 15, "budget": 4}
EXPECTED_MORSE = {"psi1": 0, "psi2": 1, "trivial": 2}

RESULTS = Path(__file__).parent / "results"


# Hessian diagnostic and Newton.
#########################################################


def grad_hessian(model, hessian=True):
    """Gradient and Hessian of Pi with respect to the free DOFs (values_reduced).

    Computed by double backward on the model's own parameter: torch.func.functional_call
    does not restore this parameter after the call (it is left as a non-leaf tensor).

    Returns:
        tuple: the parameter, grad Pi (flat), Hessian (n, n) or None.
    """
    (p,) = model.parameters()
    g = torch.autograd.grad(model(), p, create_graph=hessian)[0].flatten()
    if not hessian:
        return p, g.detach(), None
    H = torch.stack([torch.autograd.grad(gi, p, retain_graph=True)[0].flatten() for gi in g])
    return p, g.detach(), H.detach()


def diagnose(model):
    """||grad Pi||, lambda_min and number of negative eigenvalues of the Hessian."""
    _, g, H = grad_hessian(model)
    eig = torch.linalg.eigvalsh(0.5 * (H + H.T))
    return {
        "grad_norm": g.norm().item(),
        "lambda_min": eig[0].item(),
        "n_neg": int((eig < -NEG_EIG_TOL * eig.abs().max()).sum()),
    }


def newton(model, tol=NEWTON_TOL, max_iter=50):
    """Newton on grad Pi = 0, in place. Converges to saddles as well as minima.

    Returns:
        tuple: number of iterations, converged (bool).
    """
    for k in range(max_iter):
        p, g, H = grad_hessian(model)
        if g.norm() < tol:
            return k, True
        with torch.no_grad():
            p -= torch.linalg.solve(H, g).view_as(p)
    return max_iter, False


# Runs.
#########################################################


def faulty_cases():
    """Test F runs with noise that did not end on psi1, with the cause of their stop."""
    runs = pd.read_csv(RESULTS / "runs.csv")
    cases = runs[(runs["test"] == "F") & (runs["noise"] > 0) & (runs["branch"] != "psi1")].copy()
    cases["stop_cause"] = cases["converged"].map({True: "tol", False: "budget"})
    found = cases["stop_cause"].value_counts().to_dict()
    if found != N_CASES:
        sys.exit(f"Faulty cases {found} differ from the plan {N_CASES}: stopping.")
    return cases


def make_problem(init, noise, seed, n_el, quad):
    """Problem at P_F started on `init` ("trivial", "psi1" or "psi2") plus noise."""
    w_init = None if init == "trivial" else IMPERFECTIONS[init](exact_amplitude(P_F, MODES[init][1]))
    return build_problem(n_el, P_F, w_init, noise=noise, seed=seed, quadrature=QUADRATURES[quad]())


def row(test, criterion, problem, iters, converged, start, **keys):
    """One line of saddle_check.csv."""
    model = problem["model"]
    energy = model().item()
    x, w = sample_deflection(problem)
    out = classify_branch(x, w, P_F, energy)
    return {
        "test": test, **keys, "criterion": criterion,
        **{k: out[k] for k in ("a_psi1", "a_psi2", "a_psi3", "branch", "sign")},
        "energy": energy, **diagnose(model), "grad_norm_start": start["grad_norm"],
        "iters": iters, "converged": converged,
    }


def g3():
    """Exact discrete critical points from the analytical psi1, psi2 and w = 0."""
    rows = []
    for init in ["psi1", "psi2", "trivial"]:
        problem = make_problem(init, 0.0, None, 16, "gauss3")
        start = diagnose(problem["model"])
        iters, converged = newton(problem["model"])
        rows.append(row("G3", "newton", problem, iters, converged, start,
                        init=init, noise=0.0, seed=None, n_el=16, quad="gauss3", stop_cause=None))
    return rows


def g0(case):
    """The case replayed with the original criterion, diagnosed at its stop."""
    keys = dict(init=case["init"], noise=case["noise"], seed=int(case["seed"]),
                n_el=int(case["n_el"]), quad=case["quad"], stop_cause=case["stop_cause"])
    adam_iters, lbfgs_iters = OPTIMIZERS["adam+lbfgs"]
    common = dict(adam_lr=ADAM_LR, lbfgs_lr=LBFGS_LR, tol=TOL, verbose=False)
    args = (case["init"], case["noise"], keys["seed"], keys["n_el"], case["quad"])

    # G0: same path as test F
    problem = make_problem(*args)
    start = diagnose(problem["model"])
    history = minimize(problem["model"], adam_iters=adam_iters, lbfgs_iters=lbfgs_iters, **common)
    r0 = row("G0", "tol", problem, len(history["loss"]), history["converged"], start, **keys)
    r0["reproduced"] = (
        r0["branch"] == case["branch"] and r0["iters"] == case["iters"]
        and abs(r0["energy"] - case["energy"]) < 1e-10 * max(abs(case["energy"]), 1.0)
    )

    return r0


def _write(rows):
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "saddle_check.csv", index=False)
    print(f"Written to {RESULTS / 'saddle_check.csv'}")
    return out


def main():
    torch.set_num_threads(1)  # as in branch_selection.run
    cases = faulty_cases()

    print("G3: exact critical points (Newton)")
    rows = g3()
    for r in rows:
        print(f"  {r['init']:8s} ||grad|| = {r['grad_norm']:.1e}  lambda_min = {r['lambda_min']:+.3e}"
              f"  Morse index = {r['n_neg']} (expected {EXPECTED_MORSE[r['init']]})")
    if any(r["n_neg"] != EXPECTED_MORSE[r["init"]] or not r["converged"] for r in rows):
        _write(rows)
        sys.exit("G3 failed (Morse index or Newton convergence): stopping before G0.")

    print(f"G0: {len(cases)} cases")
    for _, case in cases.iterrows():
        rows.append(g0(case))
    out = _write(rows)

    replayed = out[out["test"] == "G0"]
    if not replayed["reproduced"].all():
        sys.exit(f"G0: {int(replayed['reproduced'].sum())}/{len(replayed)} reproduced.")
    print("G0: all cases reproduced")
    cols = ["init", "noise", "seed", "stop_cause", "branch", "energy", "grad_norm", "lambda_min", "n_neg", "iters"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(replayed[cols].to_string(index=False, float_format="%.2e"))


if __name__ == "__main__":
    main()
