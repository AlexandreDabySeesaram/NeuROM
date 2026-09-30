"""Branch selection by the minimization (stage 1bis).

Runs the tests A-E of the plan and writes, in `results/`:
- runs.csv: one row per run (parameters, projection on psi1/psi2/psi3, branch,
  energy error, iterations, convergence);
- trajectories.csv: projections and loss at each iteration, for the first
  `--n-traj` seeds of each configuration.

Usage:
    uv run python branch_selection.py                 # all tests
    uv run python branch_selection.py --tests ABC D   # a subset
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from main import build_problem, minimize, sample_deflection, symmetric_mode, antisymmetric_mode
from branches import classify_branch, project
from neurom.quadratures import TwoPoints1D, ThreePoints1D


P_REF = 1.5 * 4 * torch.pi**2
NOISE = 1e-8

QUADRATURES = {"gauss2": TwoPoints1D, "gauss3": ThreePoints1D}

# name -> (adam_iters, lbfgs_iters)
OPTIMIZERS = {"adam": (500, 0), "adam+lbfgs": (500, 200), "lbfgs": (0, 200)}
ADAM_LR, LBFGS_LR, TOL = 0.1, 1.0, 1e-6

IMPERFECTIONS = {"psi1": symmetric_mode, "psi2": antisymmetric_mode}


def run(cfg):
    """One minimization. Returns (row dict, list of trajectory rows)."""
    torch.set_num_threads(1)
    w_init = None
    if cfg["init"] != "noise":
        w_init = IMPERFECTIONS[cfg["init"]](cfg["amplitude"])
    problem = build_problem(
        cfg["n_el"], cfg["P"], w_init,
        noise=cfg["noise"], seed=cfg["seed"], quadrature=QUADRATURES[cfg["quad"]](),
    )

    traj = []

    def record(phase, i, loss):
        x, w = sample_deflection(problem)
        coeffs, _ = project(x, w)
        traj.append({**_keys(cfg), "step": len(traj), "phase": phase, "loss": loss,
                     **{f"a_{k}": v for k, v in coeffs.items()}})

    adam_iters, lbfgs_iters = OPTIMIZERS[cfg["optim"]]
    history = minimize(
        problem["model"], adam_lr=ADAM_LR, adam_iters=adam_iters,
        lbfgs_lr=LBFGS_LR, lbfgs_iters=lbfgs_iters, tol=TOL,
        verbose=False, callback=record if cfg["trajectory"] else None,
    )

    energy = problem["model"]().item()
    x, w = sample_deflection(problem)
    row = {
        **_keys(cfg),
        "noise": cfg["noise"],
        **classify_branch(x, w, cfg["P"], energy),
        "energy": energy,
        "iters": len(history["loss"]),
        "converged": history["converged"],
    }
    return row, traj


def _keys(cfg):
    """Columns identifying a run."""
    return {
        "test": cfg["test"], "seed": cfg["seed"], "n_el": cfg["n_el"],
        "optim": cfg["optim"], "quad": cfg["quad"],
        "P_over_4pi2": cfg["P"] / (4 * torch.pi**2),
        "init": cfg["init"], "amplitude": cfg["amplitude"],
    }


def configurations(tests, seeds, n_traj, n_elements, best_optim, best_quad):
    """All runs of the requested tests, as a list of dicts."""
    cfgs = []

    def add(test, n_el, optim, quad, P, init, amplitude, noise, seed):
        cfgs.append(dict(
            test=test, n_el=n_el, optim=optim, quad=quad, P=P, init=init,
            amplitude=amplitude, noise=noise, seed=seed,
            trajectory=noise > 0 and seed < n_traj or noise == 0 and init != "noise",
        ))

    # A, B, C: noise only, optimizer x quadrature
    if "ABC" in tests:
        for n_el, optim, quad, s in itertools.product(n_elements, OPTIMIZERS, QUADRATURES, seeds):
            add("ABC", n_el, optim, quad, P_REF, "noise", 0.0, NOISE, s)

    # D: mode imperfection, with or without noise
    if "D" in tests:
        for n_el, init, amp in itertools.product(n_elements, IMPERFECTIONS, [1e-2, 1e-4, 1e-6, 1e-8]):
            add("D", n_el, "adam+lbfgs", "gauss2", P_REF, init, amp, 0.0, 0)
            for s in seeds:
                add("D", n_el, "adam+lbfgs", "gauss2", P_REF, init, amp, NOISE, s)

    # E: load above P2, where the antisymmetric branch exists
    if "E" in tests:
        for n_el, P, s in itertools.product(n_elements, [P_REF, 10 * torch.pi**2, 30 * torch.pi**2], seeds):
            add("E", n_el, best_optim, best_quad, P, "noise", 0.0, NOISE, s)

    return cfgs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", nargs="+", default=["ABC", "D", "E"], choices=["ABC", "D", "E"])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--n-traj", type=int, default=3, help="seeds with a recorded trajectory")
    parser.add_argument("--n-elements", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--best-optim", default="adam+lbfgs", choices=list(OPTIMIZERS), help="test E")
    parser.add_argument("--best-quad", default="gauss3", choices=list(QUADRATURES), help="test E")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    cfgs = configurations(
        args.tests, range(args.seeds), args.n_traj, args.n_elements, args.best_optim, args.best_quad,
    )
    print(f"{len(cfgs)} runs on {args.workers} workers")

    rows, trajs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (row, traj) in enumerate(pool.map(run, cfgs, chunksize=4)):
            rows.append(row)
            trajs.extend(traj)
            if (k + 1) % 50 == 0 or k + 1 == len(cfgs):
                print(f"  {k + 1}/{len(cfgs)}")

    args.out.mkdir(exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(args.out / "runs.csv", index=False)
    pd.DataFrame(trajs).to_csv(args.out / "trajectories.csv", index=False)

    summary = (
        runs.assign(label=runs["branch"] + runs["sign"].map({1: "+", -1: "-", 0: ""}))
        .groupby(["test", "n_el", "optim", "quad", "P_over_4pi2", "init", "amplitude", "noise"])["label"]
        .value_counts().unstack(fill_value=0)
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
