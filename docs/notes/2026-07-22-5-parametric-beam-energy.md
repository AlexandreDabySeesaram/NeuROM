# 2026-07-22 — 5-parametric beam with tanh-graded modulus

Long-form notes behind the CHANGELOG entry of the same date.

## The example

`docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
the 1D bar now has a two-zone Young's modulus

    E(x, E1, E2, alpha, n) = (E2-E1)/2 tanh(n(x-alpha)) + (E2+E1)/2

giving five coordinates `(x, E1, E2, alpha, n)` and a CP-PGD
`u = sum_i X_i lambda_i mu_i A_i N_i`.

## Architecture

The energy is *injected* into `build_problem(loss_fn, ...)`, so the same wiring
will drive the non-linear PGD functionals unchanged. Nothing was added under
`src/` — the energy stays inline while the formulation moves.

The existing `physics.Term` contract is single-axis
(`integrand -> (N_e, N_q, 1)`) and does not fit a multi-axis separated energy;
extending it is deliberately deferred.

## The key numerical point

`tanh(n(x-alpha))` is *not* separable — it couples x, alpha and n. It is
integrated by an exact 3-D tensor-product quadrature over those axes'
quadrature points (one `einsum` per mode pair, grid `(29, 14, 14)`, rebuilt
each call so r-adaptive meshes stay correct). The `(E2 +/- E1)/2` prefactors
remain 1-D moments.

**Rejected alternatives:**

- Offline low-rank separation of `tanh` — introduces extra approximation error
  on top of the discretisation error, and would make the energy's accuracy
  depend on a separation rank nobody is tracking.
- Monte-Carlo sampling of the coupled block — noisy gradients, which the
  greedy enrichment criteria (all of which watch a loss stream for a plateau)
  are badly placed to tolerate.

## Verification

`docs/examples/1d_5-parametric_beam_PGD/tests/test_1d_5_parametric_beam_energy.py`
(17 tests at the time of writing; 18 after the training test was added
2026-07-24):

- The separated energy matches a brute-force 5-D tensor-product quadrature —
  one that exploits no separability at all — to `1e-9` in double precision, for
  1 and 2 modes and for both `MidPoint1D` and `TwoPoints1D`.
- It is differentiable w.r.t. every monom of every active mode.
- It still matches the reference after `add_mode()`, pinning the
  greedy-enrichment seam.

Two deliberate-bug injections confirmed the tests bite: a wrong-coordinate-axis
read, and a corrupted printed value.

At the 0.5 seed the script prints `energy = 6.602113e+07`, which an independent
hand computation reproduces to 5 significant figures.

## `build_problem` takes an injectable `quad`

Default `MidPoint1D`. This exists to make the `N_q > 1` path testable: one
quadrature point per element previously hid a broadcasting bug in the
2-parameter ancestor that was silently wrong at `N_q=1` and crashed at `N_q=2`.

**Test-authoring trap found here:** instantiating a quadrature rule at
`@parametrize` decoration time freezes its buffers at collection-time float32
and clashes with a float64 fixture. Parametrize over the *class*, instantiate in
the body.

## Parameter ranges

| Axis | Interval | Nodes |
|---|---|---|
| `x` | [0, 10] | 30 |
| `E1` | [10, 100] | 20 |
| `E2` | [10, 100] | 20 |
| `alpha` | [2, 8] | 15 |
| `n` | [0.5, 5] | 15 |

Note `n` never reaches 0, so the near-uniform-modulus regime is not sampled.

Mode budget set to 10 on the assumption that the non-separable modulus is far
from rank-1 — **untested**.

## Known scaling limit — the first thing that will break

The `(Qx, Qalpha, Qn)` tanh grid is built *inside* the autograd graph and
rebuilt every iteration. At the defaults that is 5 684 entries — trivial — but
~300 space elements with ~100 each in alpha and n gives a **3 M-entry
graph-retained tensor per iteration**.

Fixes, in order of effort: cache it (invalidating on mesh change), or contract
alpha/n first; and exploit the `(i,j)`/`(j,i)` symmetry the mode-pair loop
currently ignores.

**Related trap:** the grid hard-codes mode 0's quadrature points, so per-mode
independent meshes would corrupt the energy *silently* rather than crash.
