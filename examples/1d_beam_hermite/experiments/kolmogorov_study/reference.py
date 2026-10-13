"""Semi-analytical reference of the beam on a foundation under a transverse load.

Plan: `Plan-tests-barriere-kolmogorov.md` (notes), step V1. Gives a solution
computed without finite elements nor optimizer, to check that NeuROM
(`build_problem` + `minimize`) solves the problem:

    w'''' + N w'' + k w = f(x),   w = w' = 0 at x = 0 and x = 1,
    N = P - S / 2,   S = int_0^1 w'^2.

At fixed N the equation is linear; the reference is the N solving the scalar
equation N = P - S(N) / 2.
"""

import torch


torch.set_default_dtype(torch.float64)


def linear_solution(N, k, load, n_steps=4000):
    """Clamped-clamped solution of w'''' + N w'' + k w = f at fixed N.

    Use: the linear problem behind the reference; also checkable alone
    (k = 0, N = 0, f = 1 gives x^2 (1 - x)^2 / 24).

    How: first-order system y' = A y + (0, 0, 0, f) with y = (w, w', w'', w''').
    Exact propagation over each step h by E = exp(A h) (torch.linalg.matrix_exp,
    valid even when roots of r^4 + N r^2 + k coincide, unlike exponentials
    e^{r x}); the load integral over a step by 3-point Gauss-Legendre. Since
    w(0) = w'(0) = 0, the unknowns are w''(0) and w'''(0), fixed by the 2x2
    system w(1) = w'(1) = 0 (shooting, linear hence exact). The growth of
    exp(A) ~ e^beta limits it to moderate k (beta = (k/4)^(1/4) ~ 7 at k = 1e4).

    Args:
        N (float): effective axial load.
        k (float): foundation stiffness.
        load (callable): x -> f, torch function.
        n_steps (int): number of steps on [0, 1] (even, for Simpson's rule).

    Returns:
        tuple: x (n_steps + 1,) and y (n_steps + 1, 4) = (w, w', w'', w''').
    """
    h = 1.0 / n_steps
    x = torch.linspace(0.0, 1.0, n_steps + 1)
    A = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-k, 0.0, -N, 0.0],
    ])
    E = torch.linalg.matrix_exp(A * h)

    # Load contribution of each step: sum_q w_q exp(A (h - s_q)) e4 f(x_j + s_q)
    xi = torch.tensor([-(3 / 5) ** 0.5, 0.0, (3 / 5) ** 0.5])
    weights = torch.tensor([5 / 9, 8 / 9, 5 / 9]) * h / 2
    s = h / 2 * (1 + xi)
    B = torch.stack([torch.linalg.matrix_exp(A * (h - sq))[:, 3] for sq in s])  # (3, 4)
    f_q = load(x[:-1, None] + s[None, :])  # (n_steps, 3)
    d = (f_q * weights) @ B  # (n_steps, 4)

    # Columns: particular solution (zero initial state), then the responses to
    # w''(0) = 1 and to w'''(0) = 1 (no load)
    Y = torch.zeros(n_steps + 1, 4, 3)
    Y[0, 2, 1] = 1.0
    Y[0, 3, 2] = 1.0
    for j in range(n_steps):
        Y[j + 1] = E @ Y[j]
        Y[j + 1, :, 0] += d[j]

    c = torch.linalg.solve(Y[-1, :2, 1:], -Y[-1, :2, 0])
    y = Y[:, :, 0] + Y[:, :, 1:] @ c
    return x, y


def _simpson(values, x):
    """Composite Simpson's rule on a uniform grid with an even number of steps."""
    h = x[1] - x[0]
    return h / 3 * (values[0] + values[-1] + 4 * values[1:-1:2].sum() + 2 * values[2:-1:2].sum())


def reference_solution(k, P, load, n_steps=4000, tol=1e-13):
    """Reference w for the von Karman beam on a foundation, below P_cr(k).

    Use: independent check of the NeuROM solution in V1.

    How: N -> S(N) = int w_N'^2, w_N from `linear_solution`; the root of
    h(N) = N - P + S(N) / 2 is found by bisection. Below P_cr, S increases
    with N, so h is increasing and the root is unique, in [N_lo, P] with
    h(P) = S(P) / 2 > 0; N_lo is moved down until h(N_lo) < 0.

    Args:
        k (float): foundation stiffness.
        P (float): axial load, P < P_cr(k).
        load (callable): x -> f, torch function.
        n_steps (int): see `linear_solution`.
        tol (float): bisection stops when the bracket is below tol * max(1, |P|).

    Returns:
        dict: x, w, dw, ddw (n_steps + 1,), N, S and the energy
        Pi = 1/2 int w''^2 - P/2 S + S^2/8 + k/2 int w^2 - int f w.
    """
    def residual(N):
        x, y = linear_solution(N, k, load, n_steps)
        return N - P + _simpson(y[:, 1] ** 2, x).item() / 2

    N_hi = P
    gap = max(1.0, abs(P))
    N_lo = P - gap
    while residual(N_lo) > 0:
        gap *= 2
        N_lo = P - gap

    while N_hi - N_lo > tol * max(1.0, abs(P)):
        N_mid = 0.5 * (N_lo + N_hi)
        if residual(N_mid) > 0:
            N_hi = N_mid
        else:
            N_lo = N_mid
    N = 0.5 * (N_lo + N_hi)

    x, y = linear_solution(N, k, load, n_steps)
    w, dw, ddw = y[:, 0], y[:, 1], y[:, 2]
    S = _simpson(dw**2, x).item()
    energy = (
        0.5 * _simpson(ddw**2, x) - 0.5 * P * S + S**2 / 8
        + 0.5 * k * _simpson(w**2, x) - _simpson(load(x) * w, x)
    ).item()
    return {"x": x, "w": w, "dw": dw, "ddw": ddw, "N": N, "S": S, "energy": energy}
