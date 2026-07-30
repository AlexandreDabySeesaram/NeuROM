# Why the highest-degree NL term always takes the mode

**Measured**, mode 3 (first NL mode) of ledger rows `2224b67d`, `30b136ed`,
`2b605da0` in `sweep_results_new_strategies.jsonl`. Default mesh, `l3`, single
runs, all `renormalise=False`.

Each term's contribution splits as `|C_λ| · L_λ`, with `L_λ = ∏ⱼ‖wⱼ‖^λⱼ` the
norm of its basis function.

| run | λ | `L_λ/A` | `\|C\|` | share |
|---|---|---|---|---|
| uniform2 | (1,1,1,1,1) | 1 | 0.99 | 0.4 % |
| | (2,2,2,2,2) | 1.7e4 | 1.4e-2 | 99.6 % |
| uniform3 | (1,1,1,1,1) | 1 | 0.92 | 0.0 % |
| | (2,2,2,2,2) | 1.3e4 | 7.6e-2 | 0.1 % |
| | (3,3,3,3,3) | 1.7e8 | 9.8e-3 | 99.9 % |
| totaldeg6 | (1,1,1,1,1) | 1 | 1.00 | 91.8 % |
| | (2,1,1,1,1) | 4.2 | 1.2e-2 | 4.5 % |

`|C|` is flat across every row and every exponent set (5e-4 .. 8e-2). What spans
eight orders is `L_λ`.

## Mechanism

Adam already normalises the gradient: every row of `C` steps by `~lr`. The
resulting **field** motion is `lr · L_λ`. With `‖wⱼ‖ ∈ [3, 30]` (so `A ≈ 1e4`),
`L_λ` grows like `A^p` and the highest-degree row wins the race by construction,
independently of whether it fits anything.

Adding gradient normalisation is therefore the wrong fix — it is the cause. What
is unnormalised is the **basis**, not the gradient.

## Why the energy does not improve during the transfer

Every term alone is a rank-1 separated function, and with the monoms free,
`∏ⱼ wⱼ^p` spans exactly the same set as `∏ⱼ wⱼ` (take `wⱼ → wⱼ^{1/p}`). A single
correction term adds **no expressivity**; expressivity requires several terms
active at once. A 99.9 % share in one term is a mode that has fallen back to
rank-1 — CP, re-expressed with a 1e-2 coefficient against a 1e8 basis.

## Explains

| observation | leverage `max L_λ/A` |
|---|---|
| uniform3 diverges to `-1e12` | 1.7e8 |
| uniform2 "correct but small increments" | 1.7e4 |
| totaldeg6 "micro-corrections", needs `coefficient_lr` ×10 | 17 |
| `support`+totaldeg6 "stages do nothing" | ≤17, minus the pinned rows |

## Prediction under test

`renormalise=True` with `leading_coefficient=True` normalises all `d` monoms
(`polynomial_pgd.py`, `has_leading_coefficients` branch of `renormalise_mode`),
hence `L_λ = 1` for **every** λ — no row has leverage. Falsifiable: the top row's
share should fall to the same order as the others.

Queued as single-knob controls `l3-lead_coeffTrue-uniform2-r10-renorm` and
`...-uniform3-r6-renorm` in `sweep.py`. Both `joint`: `renormalise` writes in
place under `no_grad`, so under `support` it would also rescale the frozen space
monom and confound the control.
