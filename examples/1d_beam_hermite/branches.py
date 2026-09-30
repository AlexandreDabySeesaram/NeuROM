"""Identification of the buckled branch reached by a converged deflection."""

import torch

from main import symmetric_mode, antisymmetric_mode, antisymmetric_gamma


# Reference modes: name -> (unit-amplitude mode, Gamma).
MODES = {
    "psi1": (symmetric_mode(1.0, m=1), 2 * torch.pi),
    "psi2": (antisymmetric_mode(1.0, n=1), antisymmetric_gamma(1)),
    "psi3": (symmetric_mode(1.0, m=2), 4 * torch.pi),
}


def exact_amplitude(P, gamma):
    """c_n = 2 sqrt(P / Gamma_n^2 - 1), or nan below the critical load."""
    return 2 * (P / gamma**2 - 1) ** 0.5 if P > gamma**2 else float("nan")


def exact_energy(P, gamma):
    """Pi_n = -(P - Gamma_n^2)^2 / 2 on branch n, 0 below the critical load.

    On branch n, int w'^2 = 2 (P - Gamma_n^2) and int w''^2 = Gamma_n^2 int w'^2.
    """
    return -0.5 * (P - gamma**2) ** 2 if P > gamma**2 else 0.0


def _trapz_gram(x, fields):
    """Gram matrix of L2 inner products of `fields` sampled on x (trapezoidal rule)."""
    F = torch.stack(fields)
    return torch.trapezoid(F.unsqueeze(0) * F.unsqueeze(1), x, dim=-1)


def project(x, w):
    """Least-squares coefficients of w on span(psi1, psi2, psi3).

    The modes are not orthogonal (int psi1 psi3 != 0), hence a Gram solve.

    Args:
        x, w (torch.Tensor): fine sampling of the deflection, shape (N_pts,).

    Returns:
        tuple: dict name -> coefficient, relative L2 residual of the fit.
    """
    psis = [mode(x) for mode, _ in MODES.values()]
    G = _trapz_gram(x, psis)
    b = torch.stack([torch.trapezoid(p * w, x) for p in psis])
    a = torch.linalg.solve(G, b)
    residual = w - sum(ak * p for ak, p in zip(a, psis))
    norm_w = torch.trapezoid(w**2, x).sqrt()
    rel_res = (torch.trapezoid(residual**2, x).sqrt() / norm_w).item() if norm_w > 0 else 0.0
    return dict(zip(MODES, a.tolist())), rel_res


def classify_branch(x, w, P, energy, *, trivial_tol=1e-3, residual_tol=1e-1):
    """Name the branch reached by the deflection w.

    Args:
        x, w (torch.Tensor): fine sampling of w, shape (N_pts,).
        P (float): non-dimensional axial load.
        energy (float): computed Pi.
        trivial_tol (float): max |w| under which w is the trivial branch.
        residual_tol (float): relative residual above which the fit onto the
            three modes is not trusted ("other").

    Returns:
        dict: a_psi1, a_psi2, a_psi3, residual, branch ("trivial", "psi1",
        "psi2", "psi3" or "other"), sign, amplitude ratio a / c_n and
        Pi - Pi_n.
    """
    coeffs, rel_res = project(x, w)
    out = {f"a_{k}": v for k, v in coeffs.items()}
    out["residual"] = rel_res

    if w.abs().max() < trivial_tol:
        branch, sign, gamma = "trivial", 0, None
    elif rel_res > residual_tol:
        branch, sign, gamma = "other", 0, None
    else:
        # Dominant mode by its L2 contribution |a_k| ||psi_k||
        norms = {k: torch.trapezoid(m(x) ** 2, x).sqrt().item() for k, (m, _) in MODES.items()}
        branch = max(coeffs, key=lambda k: abs(coeffs[k]) * norms[k])
        sign = 1 if coeffs[branch] > 0 else -1
        gamma = MODES[branch][1]

    if gamma is None:
        out["amplitude_ratio"] = float("nan")
        out["energy_error"] = energy - 0.0 if branch == "trivial" else float("nan")
    else:
        out["amplitude_ratio"] = abs(coeffs[branch]) / exact_amplitude(P, gamma)
        out["energy_error"] = energy - exact_energy(P, gamma)
    out["branch"] = branch
    out["sign"] = sign
    return out
