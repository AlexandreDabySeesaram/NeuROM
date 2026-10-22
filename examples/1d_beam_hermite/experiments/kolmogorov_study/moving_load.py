"""Moving localized load on a von Karman beam resting on a Winkler foundation.

Plan: `Plan-tests-barriere-kolmogorov.md` (notes). Problem-specific helpers of
the Kolmogorov-barrier study; the energy itself is `main.build_problem` with
the `k` and `load` arguments:
- `bspline_load`: the transverse load density F b_s(x - a);
- `linear_stiffness`: K_0, Hessian of Pi at w = 0, P = 0 (preconditioner);
- `membrane_stiffness`: G, Hessian of 1/2 int w'^2;
- `buckling_modes`: the buckling loads P_n(k) and modes phi_n (`generalized_modes`
  from the assembled matrices);
- `critical_load`: P_cr(k) and the critical mode (first of `buckling_modes`);
- `load_scale`: the intensity F(k) of the "strong" load.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from neurom.quadratures import FourPoints1D

from main import build_problem
from utils import grad_hessian


def bspline_load(F, s, a):
    """Transverse load density x -> F b_s(x - a), of total intensity F.

    Use: the moving load of the study, passed as `load=` to `build_problem`.
    b_s(xi) = B(xi / s) / s with B the quadratic B-spline (unit area, standard
    deviation s / 2): C1 and piecewise quadratic on [-3s/2, 3s/2]. When a is a
    node and s = 2h (h the element size), its breakpoints are nodes, so the
    Hermite interpolation of the load is exact and, with FourPoints1D, so is
    int f w (degree 6 per element).

    Returns:
        callable: x -> F b_s(x - a).
    """
    def load(x):
        t = torch.abs((x - a) / s)
        return F * (0.5 * torch.clamp(1.5 - t, min=0) ** 2 - 1.5 * torch.clamp(0.5 - t, min=0) ** 2) / s
    return load


def linear_stiffness(k, n_elements, quadrature=None):
    """Linear stiffness K_0: Hessian of Pi on the free DOFs at w = 0 and P = 0.

    Use: K_0 comes from 1/2 int w''^2 + k/2 int w^2, it is symmetric positive
    definite for every P; its Cholesky factor preconditions LBFGS
    (`utils.cholesky_preconditioner`) and it is the K of `critical_load`.

    Args:
        k (float): foundation stiffness.
        n_elements (int): number of Hermite elements.
        quadrature (QuadratureRule, optional): defaults to FourPoints1D().

    Returns:
        torch.Tensor: K_0, shape (n, n), symmetrized.
    """
    quadrature = FourPoints1D() if quadrature is None else quadrature
    _, _, K = grad_hessian(build_problem(n_elements, 0.0, k=k, quadrature=quadrature)["model"])
    return 0.5 * (K + K.T)


def membrane_stiffness(n_elements, quadrature=None):
    """G: Hessian of 1/2 int w'^2 on the free DOFs (independent of k and P).

    How: at w = 0 the Hessian of Pi is H(P) = K_0 - P G (the membrane term is
    quartic, so it does not contribute), hence G = H(0) - H(1), from two
    `build_problem` calls.

    Args:
        n_elements (int): number of Hermite elements.
        quadrature (QuadratureRule, optional): defaults to FourPoints1D().

    Returns:
        torch.Tensor: G, shape (n, n), symmetrized.
    """
    quadrature = FourPoints1D() if quadrature is None else quadrature
    K = linear_stiffness(0.0, n_elements, quadrature)
    _, _, H1 = grad_hessian(build_problem(n_elements, 1.0, quadrature=quadrature)["model"])
    return 0.5 * ((K - H1) + (K - H1).T)


def buckling_modes(k, n_elements, n_modes, quadrature=None):
    """First buckling loads P_n(k) and modes phi_n of the beam on its foundation.

    Use: the branches of the unloaded beam (w = c_n phi_n on branch n) and the
    directions +-phi_n from which the minimizations start above P_cr.

    How: at w = 0 the Hessian of Pi on the free DOFs is H(P) = K - P G, with
    K = `linear_stiffness` and G = `membrane_stiffness`. The P_n are the
    smallest eigenvalues of K phi = P G phi, solved as a symmetric problem
    through the Cholesky factor of G.

    Args:
        k (float): foundation stiffness.
        n_elements (int): number of Hermite elements.
        n_modes (int): number of modes.
        quadrature (QuadratureRule, optional): defaults to FourPoints1D().

    Returns:
        tuple: P_n (n_modes,) in increasing order, phi_n on the free DOFs
        (n, n_modes), each normalized to max |w| = 1 over the nodes, positive
        at its extremum.
    """
    K = linear_stiffness(k, n_elements, quadrature)
    G = membrane_stiffness(n_elements, quadrature)
    return generalized_modes(K, G, n_modes)


def generalized_modes(K, G, n_modes):
    """First eigenpairs of K phi = P G phi (the computation of `buckling_modes`).

    Use: called directly with K = K_b + k M when many k are needed, K_b, M and
    G being assembled once (K_0 is affine in k).

    How: symmetric problem through the Cholesky factor of G.

    Returns:
        tuple: P_n (n_modes,) in increasing order, phi_n on the free DOFs
        (n, n_modes), each normalized to max |w| = 1 over the nodes, positive
        at its extremum.
    """
    L = torch.linalg.cholesky(G)
    A = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, K, upper=False).T, upper=False)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (A + A.T))
    phi = torch.linalg.solve_triangular(L.T, eigvecs[:, :n_modes], upper=True)

    # Free DOFs are interleaved (w, w') once the clamped ends are removed
    w_nodes = phi[0::2]
    phi = phi / w_nodes.gather(0, w_nodes.abs().argmax(0, keepdim=True))
    return eigvals[:n_modes], phi


def critical_load(k, n_elements, quadrature=None):
    """Critical load P_cr(k) of the beam on its foundation, and the critical mode.

    Use: sets the load levels of the study (P / P_cr) and gives the direction
    +-phi_cr from which the minimizations start above P_cr. First mode of
    `buckling_modes`.

    Args:
        k (float): foundation stiffness.
        n_elements (int): number of Hermite elements.
        quadrature (QuadratureRule, optional): defaults to FourPoints1D().

    Returns:
        tuple: P_cr (float), phi_cr on the free DOFs (n,), normalized to
        max |w| = 1 over the nodes, positive at its extremum.
    """
    P_n, phi = buckling_modes(k, n_elements, 1, quadrature)
    return P_n[0].item(), phi[:, 0]


def load_scale(k, s, n_elements, w_target=1.0, quadrature=None):
    """Intensity F of the B-spline load giving max |w| = w_target at P = 0, a = 0.5.

    Use: the foundation stiffens the beam by orders of magnitude when k grows,
    so at a fixed F the deflection, hence the weight of the membrane term
    1/8 (int w'^2)^2, would change with k. Scaling F so that max |w| ~ 1 (the
    radius of gyration, w = w_hat / r) keeps the same "strong load" level of
    nonlinearity for every k.

    How: linear problem (P = 0, membrane term dropped) with F = 1. At w = 0,
    grad Pi = -b (load vector) and the Hessian is K, so w = -K^-1 grad Pi; then
    F = w_target / max |w| over the nodes.

    Args:
        k (float): foundation stiffness.
        s (float): width of the B-spline.
        n_elements (int): number of Hermite elements.
        w_target (float): target max |w| of the linear response.
        quadrature (QuadratureRule, optional): defaults to FourPoints1D().

    Returns:
        float: F.
    """
    quadrature = FourPoints1D() if quadrature is None else quadrature
    problem = build_problem(n_elements, 0.0, k=k, load=bspline_load(1.0, s, 0.5), quadrature=quadrature)
    _, g, K = grad_hessian(problem["model"])
    w = -torch.linalg.solve(K, g)
    # Free DOFs are interleaved (w, w') once the clamped ends are removed
    return w_target / w[0::2].abs().max().item()
