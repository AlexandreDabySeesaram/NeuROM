## 2026-07-22 — 5-parametric beam with tanh-graded modulus

- **New example** `docs/examples/1d_5-parametric_beam_PGD/1d_5-parametric_beam_deflection_PGD.py`:
  the 1D bar now has a two-zone Young's modulus
  `E(x, E1, E2, alpha, n) = (E2-E1)/2 tanh(n(x-alpha)) + (E2+E1)/2`, giving five
  coordinates `(x, E1, E2, alpha, n)` and a CP-PGD
  `u = sum_i X_i lambda_i mu_i A_i N_i`.
- **Architecture:** the energy is *injected* into `build_problem(loss_fn, ...)`, so the
  same wiring will drive the non-linear PGD functionals unchanged. Nothing added under
  `src/` — the energy stays inline while the formulation moves. The existing
  `physics.Term` contract is single-axis (`integrand -> (N_e, N_q, 1)`) and does not fit
  a multi-axis separated energy; extending it is deliberately deferred.
- **Key numerical point:** `tanh(n(x-alpha))` is *not* separable — it couples x, alpha
  and n. It is integrated by an exact 3-D tensor-product quadrature over those axes'
  quadrature points (one `einsum` per mode pair, grid `(29, 14, 14)`, rebuilt each call
  so r-adaptive meshes stay correct). The `(E2 +/- E1)/2` prefactors remain 1-D moments.
  Rejected: offline low-rank separation of `tanh` (extra approximation error) and
  Monte-Carlo sampling (noisy gradients).
- **Verified:** the separated energy matches a brute-force 5-D tensor-product quadrature
  to 1e-9 in double precision, for 1 and 2 modes, and is differentiable w.r.t. every
  monom (`tests/integration/test_1d_5_parametric_beam_energy.py`).
- **Not yet done:** training, plotting, and a reference solution. Mode budget set to 10
  on the assumption that the non-separable modulus is far from rank-1 — untested.
- Parameter ranges: `x in [0,10]` (30 nodes), `E1, E2 in [10,100]` (20),
  `alpha in [2,8]` (15), `n in [0.5,5]` (15). Note `n` never reaches 0, so the
  near-uniform-modulus regime is not sampled.
