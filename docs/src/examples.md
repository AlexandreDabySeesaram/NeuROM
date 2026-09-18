# Examples

Runnable examples live in the [`examples/`](https://github.com/AlexandreDabySeesaram/NeuROM/tree/master/examples)
folder at the root of the repository, outside this documentation. Each one is a
self-contained directory with a `main.py` and a `README.md` walking through the
problem it solves.

This page lists the examples we chose to surface in the documentation; the
folder holds others (benchmarks and proofs of concept) that are not meant to be
read as tutorials.

## 1D parametric bar deflection with CP-PGD

[`examples/1d_beam_deflection_pgd/`](https://github.com/AlexandreDabySeesaram/NeuROM/tree/master/examples/1d_beam_deflection_pgd)

A bi-clamped 1D bar under a constant axial load, solved for **every** Young's
modulus at once: the modulus $E$ is treated as an extra coordinate and the
displacement is sought as a separated sum

$$u(x, E) = \sum_m S_m(x)\, g_m(E),$$

built greedily one mode at a time with the Proper Generalized Decomposition.
The write-up derives the separated energy, maps each factor onto the library
components, and compares the result against the analytical solution
$u = \tfrac{1}{2} f (x - x_\text{min})(x - x_\text{max}) / E$.
