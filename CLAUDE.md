This project aims to enrich the example in `docs/examples/1d_2-parametric_beam_PGD/` (`1d_beam_deflection_PGD.py`, `1d_beam_deflection_PGD.md`) with more parameters for the solution and then change the decomposition from classical CPPGD already present in the library to non-linear PGD decompositions.

Testing is done in `docs/examples/1d_5-parametric_beam_PGD/` — that is where the enriched, 5-parameter case lives (x, E_1, E_2, \alpha, n) and where the new decompositions get exercised.

The method used is the Proper Generalized Decomposition. **This is an a priori method of model order reduction**, meaning no solutions are computed and compressed to create the approximation space: the approximation space is constructed on the fly. It is already implemented in the library in `src/neurom/decompositions/pgd.py`, and described in Nouy (2010).

Instead of classically obtaining the modes via alternate directions power iterations (Ammar et al., 2006). We get the mode via a global minimization thanks to the reformulation of the FEM problem in Škardová et al. (2026), Part I.

We are building on top of Daby-Seesaram et al. (2026), Part II.

In the classical CPPGD, the solution is looked for in this general manner : $$u(\{x_i\}_{i=1,...,l}) = \sum_{j=1}^m u_j(\{x_i\}_{i=1,...,l})= \sum_{j=1}^m\prod_{k=1}^l w_j^k(x_k)$$.
We will refer to $w_j^k$ as "monoms" and the $u_j$ as "modes".
The non-linear PGD decompositions we will explore are described in /Users/solal/Library/CloudStorage/GoogleDrive-solal21a@gmail.com/Mon Drive/Solal_PhD/PhD_Solal_vault/5_Projects/NL PGD/premiers_tests/NL_PGD_rundown_4_claude.md . For now, the new decompositions go directly into `src/neurom/decompositions/pgd.py`, alongside the existing `CPPGD` class, rather than into separate modules.

You need to stick to the current library code as much as possible to respect its philosophy, if a feature is needed, first look through the library to see if something akin exists before recoding a pre-existing tool. Use dependency injection as much as possible to keep the modules portable and reusable while easy to enrich later.

## References

- Ammar, A., Mokdad, B., Chinesta, F., & Keunings, R. (2006). A new family of solvers for some classes of multidimensional partial differential equations encountered in kinetic theory modeling of complex fluids. _Journal of Non-Newtonian Fluid Mechanics_, _139_(3), 153–176. https://doi.org/10.1016/j.jnnfm.2006.07.007
- Nouy, A. (2010). A priori model reduction through Proper Generalized Decomposition for solving time-dependent partial differential equations. _Computer Methods in Applied Mechanics and Engineering_, _199_(23), 1603–1626. https://doi.org/10.1016/j.cma.2010.01.009
- Škardová, K., Daby-Seesaram, A., & Genet, M. (2026). Finite Element Neural Network Interpolation. Part I: Interpretable and Adaptive Discretization for Solving PDEs. _Computational Mechanics_, _77_(2), 547–567. https://doi.org/10.1007/s00466-025-02677-3
- Daby-Seesaram, A., Škardová, K., & Genet, M. (2026). Finite Element Neural Network Interpolation. Part II: Hybridisation with the Proper Generalised Decomposition for non-linear surrogate modelling. _Computational Mechanics_, _77_(2), 521–546. https://doi.org/10.1007/s00466-025-02676-4
## Changelog

This is a **testing branch**, not just an extension. `CHANGELOG.md` therefore
records both:

- **Architecture changes** to the library — new features, API/behavior changes,
  structural refactors, and bug fixes in `src/`.
- **Experimental status** — what was tried, what works and what does not.
  Record which decompositions/configurations converge, which fail and how they
  fail (divergence, NaNs, poor accuracy), the test or script that shows it, and
  any known-good settings. Negative results matter: write them down so they are
  not retried blindly.

Still keep it out of the log: trivial comment tweaks, formatting, and
moment-to-moment debugging churn that leads nowhere conclusive. An entry should
teach something to the next session.

Append entries to `CHANGELOG.md` at the repo root before ending the session.
Newest entries go on top. Keep each entry short — what changed or what was
tested, and the outcome — and link a fuller notes doc when the change is large.
At the start of a session, read `CHANGELOG.md` to catch up on prior work and on
the current state of what works.

**NEVER add Claude as co-author on the commits.**
