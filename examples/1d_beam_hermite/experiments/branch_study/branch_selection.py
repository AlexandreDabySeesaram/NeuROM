"""Branch selection by the minimization (stage 1bis).

Runs the tests A-F of the plan and writes, in `results/`:
- runs.csv: one row per run (parameters, projection on psi1/psi2/psi3, branch,
  energy error, iterations, convergence);
- trajectories.csv: projections and loss at each iteration, for the first
  `--n-traj` seeds of each configuration.
Rows of the tests not requested are kept from existing files.

Usage:
    uv run python branch_selection.py                 # all tests
    uv run python branch_selection.py --tests ABC D   # a subset
    uv run python branch_selection.py --tests F       # adds F to the existing results
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from main import build_problem
from utils import minimize, sample_deflection
from modes import (
    MODES, classify_branch, exact_amplitude, project, symmetric_mode, antisymmetric_mode,
)
from neurom.quadratures import TwoPoints1D, ThreePoints1D


P_REF = 1.5 * 4 * torch.pi**2
NOISE = 1e-8

QUADRATURES = {"gauss2": TwoPoints1D, "gauss3": ThreePoints1D}

# name -> (adam_iters, lbfgs_iters)
OPTIMIZERS = {"adam": (500, 0), "adam+lbfgs": (500, 200), "lbfgs": (0, 200)}
ADAM_LR, LBFGS_LR, TOL = 0.1, 1.0, 1e-6

IMPERFECTIONS = {"psi1": symmetric_mode, "psi2": antisymmetric_mode}

# Test F: start on a branch at a load where trivial, psi1 (stable) and psi2
# (unstable) all exist, and increase the noise.
P_F = 10 * torch.pi**2
NOISES_F = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
SEEDS_F = 20


def run(cfg):
    """One minimization. Returns (row dict, list of trajectory rows)."""
    torch.set_num_threads(1)
    w_init = None
    if cfg["init"] not in ("noise", "trivial"):
        w_init = IMPERFECTIONS[cfg["init"]](cfg["amplitude"])
    problem = build_problem(
        cfg["n_el"], cfg["P"], w_init,
        noise=cfg["noise"], seed=cfg["seed"], quadrature=QUADRATURES[cfg["quad"]](),
    )

    x, w = sample_deflection(problem)
    a_psi1_init = project(x, w)[0]["psi1"]
    # Branch of the unperturbed start ("noise" and "trivial" both start at w = 0)
    start = ("trivial", 0) if cfg["init"] in ("noise", "trivial") else (cfg["init"], 1)

    traj = []
    exit_step = None

    def record(phase, i, loss):
        nonlocal exit_step
        x, w = sample_deflection(problem)
        out = classify_branch(x, w, cfg["P"], loss)
        if exit_step is None and _label(out) != start:
            exit_step = i_step[0]
        if cfg["trajectory"]:
            traj.append({**_keys(cfg), "step": i_step[0], "phase": phase, "loss": loss,
                         **{k: out[k] for k in ("a_psi1", "a_psi2", "a_psi3")}})
        i_step[0] += 1

    i_step = [0]
    track = cfg["trajectory"] or cfg["test"] == "F"

    adam_iters, lbfgs_iters = OPTIMIZERS[cfg["optim"]]
    history = minimize(
        problem["model"], adam_lr=ADAM_LR, adam_iters=adam_iters,
        lbfgs_lr=LBFGS_LR, lbfgs_iters=lbfgs_iters, tol=TOL,
        verbose=False, callback=record if track else None,
    )

    params = [p for p in problem["model"].parameters() if p.requires_grad]
    loss = problem["model"]()
    grad_norm = torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)]).norm().item()
    energy = loss.item()
    x, w = sample_deflection(problem)
    row = {
        **_keys(cfg),
        "noise": cfg["noise"],
        **classify_branch(x, w, cfg["P"], energy),
        "energy": energy,
        "iters": len(history["loss"]),
        "converged": history["converged"],
        "grad_norm": grad_norm,
        "a_psi1_init": a_psi1_init,
        "exit_step": exit_step if track else None,
    }
    return row, traj


def _label(out):
    """Branch and sign, e.g. ("psi1", -1)."""
    return out["branch"], out["sign"]


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

    # F: start on the trivial, psi1 or psi2 branch at P_F, increasing noise
    if "F" in tests:
        for init in ["trivial", "psi1", "psi2"]:
            amp = 0.0 if init == "trivial" else exact_amplitude(P_F, MODES[init][1])
            add("F", 16, "adam+lbfgs", "gauss3", P_F, init, amp, 0.0, 0)
            for noise, s in itertools.product(NOISES_F, range(SEEDS_F)):
                add("F", 16, "adam+lbfgs", "gauss3", P_F, init, amp, noise, s)

    return cfgs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", nargs="+", default=["ABC", "D", "E"], choices=["ABC", "D", "E", "F"])
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
    runs = _merge(args.out / "runs.csv", pd.DataFrame(rows), args.tests)
    runs.to_csv(args.out / "runs.csv", index=False)
    _merge(args.out / "trajectories.csv", pd.DataFrame(trajs), args.tests).to_csv(
        args.out / "trajectories.csv", index=False)

    summary = (
        runs.assign(label=runs["branch"] + runs["sign"].map({1: "+", -1: "-", 0: ""}))
        .groupby(["test", "n_el", "optim", "quad", "P_over_4pi2", "init", "amplitude", "noise"])["label"]
        .value_counts().unstack(fill_value=0)
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary)
    print(f"Written to {args.out}")


def _merge(path, new, tests):
    """Replace, in the CSV at `path` if any, the rows of the tests just run."""
    if not path.exists():
        return new
    old = pd.read_csv(path)
    return pd.concat([old[~old["test"].isin(tests)], new], ignore_index=True)


if __name__ == "__main__":
    main()
