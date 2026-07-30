# The coefficient scale problem, and why `staged` does not train

Companion to the CHANGELOG entry of the same date. Everything below is measured
on `docs/examples/1d_5-parametric_beam/NLPGD/`, tiny mesh
(`space 8, E1 5, E2 5, alpha 5, n 5`), `max_iter=80, min_iter=60`, 3 modes,
`I = uniform(5, 3)`. Tiny-mesh single runs, **not** an ablation at production
size — the mechanism generalises, the numbers do not.

## The scale of a coefficient is set by the mode amplitude

Write `A = prod_j ||w_ij||` for mode `i`'s amplitude. The leading term
contributes `~A` to the field; the term with a uniform exponent `p` contributes
`C_p * A^p`. For the correction to be a *correction* rather than a takeover,

    C_p  ~  A^(1 - p)

After a converged CP stage on this problem `A ~ 1.5e4`, so `C_2 ~ 7e-5` and
`C_3 ~ 5e-9` — **four orders of magnitude apart, inside a single coefficient
row**. Measured directly after a converged CP stage (`renormalise` on, so the
amplitude sits on axis 0):

| quantity | value |
|---|---|
| per-axis monom norms | `(2.9e5, 1, 1, 1, 1)` |
| `prod ||w||^lambda`, leading | `2.9e5` |
| `prod ||w||^lambda`, `lambda = 2` | `8.4e10` |
| `prod ||w||^lambda`, `lambda = 3` | `2.5e16` |
| `dE/dC` | `(-1.6e12, -3.5e15)` |

## This is not a gauge artefact

`renormalise` cannot help. For a uniform exponent row the term is *exactly*
gauge-invariant, because the gauge orbit requires `prod_j s_j = 1`:

    prod_j (s_j w_j)^p  =  (prod_j s_j)^p prod_j w_j^p  =  prod_j w_j^p

So no redistribution of amplitude across axes changes anything. `A` is physical.
(Non-uniform rows are *not* invariant, so a `total_degree` set would interact
with the gauge — untested.)

## Adam makes it worse, not better

Adam's step is `lr * m / sqrt(v) ~ lr` **regardless of the gradient's
magnitude**. So a shared `lr = 0.1` moves `C` by ~0.1 on its first step —
five to ten orders past the target — and the huge `dE/dC` above buys nothing.
Hence `RunConfig.coefficient_lr` and the two-param-group factory in
`build_optimizer_factory`.

## The result: `joint` trains, `staged` does not

Final energy after 3 modes (lower is better — this is a minimisation, so the
comparison is legitimate). CP baseline (`greedy`, coefficients never released):
**`-1.914e11`**.

| `lr_C` | `staged` | `joint` |
|---|---|---|
| 1e-3 | `+2.65e20` diverged | **`-1.271e12`** |
| 1e-5 | `-1.888e11` ~ CP | `-2.054e11` |
| 1e-7 | `-1.923e11` ~ CP | `-1.918e11` ~ CP |
| 1e-9 | `-1.887e11` ~ CP | `-1.894e11` ~ CP |
| 1e-11 | `-1.917e11` ~ CP | `-1.893e11` ~ CP |

**There is no working `lr_C` for `staged`.** One step too big and the `p = 3`
row blows the field up; one step smaller and the `p = 2` row never leaves zero.
The gap has nothing in it. `joint` at `1e-3` reaches 6.6x lower energy than CP
at the same rank.

## Why `joint` escapes it

`staged` releases `C` against a **converged** mode, i.e. at maximal `A`, where
the two rows' target scales are four orders apart. `joint` releases `C` at the
mode's *seed*, where `A` is small (the monoms start at `0.5` / `seed_amplitude`)
and both rows' targets are within reach of one `lr`; `A` and `C` then grow
together, and `C` is never asked to jump orders of magnitude.

So the comparison "joint vs staged" is, as currently parameterised, not a
comparison of two viable schedules. It is a demonstration that the coefficient
parameterisation is amplitude-dependent.

## What would make `staged` viable

Reparameterise the coefficients against the amplitude inside
`PolynomialNLPGD`: store `C~` and evaluate

    C_lambda  =  C~_lambda * A^(1 - sum_j lambda_j / d)

so every row's natural scale is `O(1)` and one `lr` serves all of them. This
also removes the dependence of a *trained* coefficient on the mode's amplitude,
which is the same reason `renormalise` exists for the monoms.

**Not done.** It changes the decomposition's state_dict semantics (`C~` is not
`C`), so every checkpoint written before it would be misread rather than
rejected — it needs a format version, and it needs deciding whether `A` is
recomputed live or frozen per stage.
