# References

- Ammar, A., Mokdad, B., Chinesta, F., & Keunings, R. (2006). A new family of solvers for some classes of multidimensional partial differential equations encountered in kinetic theory modeling of complex fluids. _Journal of Non-Newtonian Fluid Mechanics_, _139_(3), 153–176. https://doi.org/10.1016/j.jnnfm.2006.07.007
- Nouy, A. (2010). A priori model reduction through Proper Generalized Decomposition for solving time-dependent partial differential equations. _Computer Methods in Applied Mechanics and Engineering_, _199_(23), 1603–1626. https://doi.org/10.1016/j.cma.2010.01.009
- Škardová, K., Daby-Seesaram, A., & Genet, M. (2026). Finite Element Neural Network Interpolation. Part I: Interpretable and Adaptive Discretization for Solving PDEs. _Computational Mechanics_, _77_(2), 547–567. https://doi.org/10.1007/s00466-025-02677-3
- Daby-Seesaram, A., Škardová, K., & Genet, M. (2026). Finite Element Neural Network Interpolation. Part II: Hybridisation with the Proper Generalised Decomposition for non-linear surrogate modelling. _Computational Mechanics_, _77_(2), 521–546. https://doi.org/10.1007/s00466-025-02676-4

## What each one is used for

- **Nouy (2010)** — the PGD formulation implemented as `CPPGD` in
  `src/neurom/decompositions/pgd.py`. Source of the *a priori* framing.
- **Ammar et al. (2006)** — the classical alternating-direction power-iteration
  solver. This project deliberately does **not** use it; it is the baseline
  being replaced.
- **Škardová et al. (2026), Part I** — the FEM reformulation that lets a mode be
  obtained by global minimisation instead. This is what `PGDTrainer` minimises.
- **Daby-Seesaram et al. (2026), Part II** — FENNI × PGD for non-linear
  surrogate modelling; the work this branch builds on top of.
