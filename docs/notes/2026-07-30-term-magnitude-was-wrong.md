# `term_magnitudes` was wrong: `||w^p|| != ||w||^p`

**Every term share reported before 2026-07-30 overstated the non-linear
corrections.** The diagnostic computed

    |coeff| * prod_j ||w_mj||^lambda_j

but a term's L2 norm over the product domain is, by Fubini,

    |coeff| * prod_j ||psi_{lambda_j}(w_mj)||

and the quadrature does not commute with the power. The two agree **only** for
the leading term, where every exponent is 1 — which is why the error went
unnoticed: the one row that was right was the reference everything else was
compared against.

## The size of it

Measured on `3f5a4afe`, mode 3, after `renormalise` put every monom at unit
norm — so `||w||^2` is exactly 1 on every axis:

| axis | `\|\|w\|\|²` (used) | `\|\|w²\|\|` (true) |
|---|---|---|
| space | 1.000 | 0.525 |
| E1 | 1.000 | 0.277 |
| E2 | 1.000 | 0.126 |
| alpha | 1.000 | 0.861 |
| n | 1.000 | 0.479 |

Product over the five axes: a factor **~130**, all of it inflating the
correction. `factor_norm(m, k, 1)` reproduces `monom_norms(m)[k]` exactly, which
is the check that the new form is the same quantity where they should agree.

## Leading-term share, as reported and as it is

| run | reported | exact |
|---|---|---|
| uniform2 r10 renorm | 2.3 – 18.8 % | **80.6 – 98.2 %** |
| uniform2 r10 renorm+orth | 18.4 – 55.7 % | **97.1 – 99.9 %** |
| uniform3 r6 renorm | 0.0 – 2.9 % | **81.9 – 93.4 %** |
| uniform3 r6 renorm+orth | 0.1 – 0.7 % | **97.3 – 99.6 %** |
| uniform3 r6 **no** renorm (diverged) | 0.0 – 2.9 % | 3.4 – 93.1 % |

## What this falsifies

- **"The linear term collapses."** It does not, in any renormalised run — it
  holds 80 % or more. The only run where the collapse is real is the one that
  diverged, without `renormalise`.
- **"`(3,3,3,3,3)` holds 98–100 % of every mode."** No: the leading term holds
  81.9–93.4 % there.
- **"A mode at 99 % in one correction is CP re-expressed with a tiny coefficient
  against a huge basis."** The premise was the artefact.
- **The evidence for correction↔correction redundancy.** The argument was "two
  rows fight over 99 % of the mode". That observation does not exist. Redundancy
  may still be real — the *reparameterisation* argument (`w -> w^{1/p}` makes
  `prod w^p` span the same rank-1 set as `prod w`) is independent of any
  measurement — but it is no longer supported by this one.

What is left in its place is the opposite problem: the corrections are **small**.
That is closer to the `support` finding of the same day, where `C` could not
leave 0 because `renormalise` only ever *multiplies* it.

## What survives

Everything measured through the error or the energy, which never touched this
diagnostic:

- `renormalise=True` kills the `uniform3` divergence: 2.092e+00 -> 8.811e-02
  overall, energy from -5.16e11 back to -2.039e11, inside the band.
- `orthogonal_corrections` on `uniform2 r10`: 4.232e-02 -> 3.864e-02 overall,
  worst point 1.523e-01 -> 1.140e-01.
- The `support` schedule cannot activate its correction under `renormalise`
  (`|C|` ~ 5e-3 against `|c|` ~ 1.6e4): that came from raw coefficients and the
  Adam step budget, not from shares.
- The ~6 % single-run noise floor on overall error.

## Fixed in

`PolynomialNLPGD.factor_norm(m, k, power)` integrates the factor itself; the
example's `term_magnitudes` calls it. It is also the only form that means
anything once the exponents index a non-monomial family, where `||psi_p(w)||`
has no relation to `||w||` at all.
