"""Minimization and post-processing helpers for the von Karman beam."""

import torch

from neurom.interpolation import PointWiseInterpolator


# Minimization.
#########################################################


def minimize(
    model, *, adam_lr, adam_iters, lbfgs_lr, lbfgs_iters, tol, verbose=True, callback=None,
):
    """Minimize the model energy with Adam, then LBFGS (strong Wolfe).

    Each phase stops after its number of iterations or when the relative change
    of the loss |L_k - L_{k-1}| / |L_k| falls below `tol`. A phase with 0
    iterations is skipped. One LBFGS iteration is one `optimizer.step`.

    Args:
        callback (callable, optional): called as callback(phase, i, loss)
            after each step.

    Returns:
        dict: "loss" (list of floats), "phase" (list of "adam" / "lbfgs") and
        "converged" (bool, the last phase run stopped on `tol`).
    """
    history = {"loss": [], "phase": [], "converged": False}

    phases = [
        ("adam", adam_iters, lambda: torch.optim.Adam(model.parameters(), lr=adam_lr)),
        ("lbfgs", lbfgs_iters, lambda: torch.optim.LBFGS(
            model.parameters(), lr=lbfgs_lr, line_search_fn="strong_wolfe")),
    ]

    for name, n_iters, make_optimizer in phases:
        if n_iters == 0:
            continue
        optimizer = make_optimizer()

        def closure():
            optimizer.zero_grad()
            loss = model()
            if loss.requires_grad:
                loss.backward()
            return loss

        previous = None
        history["converged"] = False
        for i in range(n_iters):
            loss = optimizer.step(closure).item()
            history["loss"].append(loss)
            history["phase"].append(name)
            if verbose:
                print(f"{name} i = {i} loss={loss:.6e}")
            if callback is not None:
                callback(name, i, loss)
            if previous is not None and abs(loss - previous) <= tol * max(abs(loss), 1e-30):
                history["converged"] = True
                break
            previous = loss

    return history


def grad_hessian(model, hessian=True):
    """Gradient and Hessian of the energy with respect to the free DOFs.

    Use: second-order information on Pi without assembling any matrix by hand,
    e.g. to classify a stationary point (sign of the eigenvalues), to run Newton,
    or to read the stiffness matrices of the linearized problem at w = 0.

    How: double backward on the model's single parameter (`values_reduced`, the
    DOFs left free by the Dirichlet conditions). torch.func.functional_call is
    not used: it does not restore this parameter after the call.

    Args:
        model (FEMModel): model whose forward returns Pi.
        hessian (bool): if False, only the gradient is computed.

    Returns:
        tuple: the parameter, grad Pi (flat, n), Hessian (n, n) or None.
    """
    (p,) = model.parameters()
    g = torch.autograd.grad(model(), p, create_graph=hessian)[0].flatten()
    if not hessian:
        return p, g.detach(), None
    H = torch.stack([torch.autograd.grad(gi, p, retain_graph=True)[0].flatten() for gi in g])
    return p, g.detach(), H.detach()


def newton(model, tol=1e-9, max_iter=50):
    """Newton iterations on grad Pi = 0, updating the model DOFs in place.

    Use: reach an exact discrete stationary point, independently of the
    stopping criteria of Adam / LBFGS (e.g. to separate optimizer error from
    discretization error). Converges to saddles as well as minima.

    Args:
        model (FEMModel): model whose forward returns Pi.
        tol (float): stop when ||grad Pi|| < tol (float64 round-off floor ~1e-10).
        max_iter (int): maximum number of Newton steps.

    Returns:
        tuple: number of iterations, converged (bool).
    """
    for k in range(max_iter):
        p, g, H = grad_hessian(model)
        if g.norm() < tol:
            return k, True
        with torch.no_grad():
            p -= torch.linalg.solve(H, g).view_as(p)
    return max_iter, False


# Post-processing.
#########################################################


def axial_displacement(stretch, field_layout, x_nodes, P):
    """Non-dimensional axial displacement at the mesh nodes.

    u(x_j) = -[(P - S/2) x_j + 1/2 int_0^{x_j} w'^2], with S = int_0^1 w'^2,
    so that u_hat = (r^2 / l) * u. The integrals reuse the quadrature of
    `stretch` (integrand w'^2 dx), i.e. the one the solver minimized.

    Args:
        stretch (SolidElasticEnergy): term with integrand w'^2 dx.
        field_layout (FieldLayout): layout holding the up-to-date quadrature results.
        x_nodes (torch.Tensor): non-dimensional node positions, shape (N_nodes,).
        P (float): non-dimensional axial load.

    Returns:
        torch.Tensor: non-dimensional axial displacement, shape (N_nodes,).
    """
    # Per-element integrals of w'^2, then cumulative from x = 0
    I_e = stretch.integrand(field_layout).detach().reshape(x_nodes.shape[0] - 1, -1).sum(dim=1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(I_e, dim=0)])
    S = cum[-1]

    return -((P - 0.5 * S) * x_nodes + 0.5 * cum)


def sample_deflection(problem, n_per_element=20):
    """Evaluate w with the Hermite shape functions on a fine grid.

    Returns:
        tuple: x (N_pts,) and w (N_pts,), non-dimensional.
    """
    n_elements = problem["x_nodes"].shape[0] - 1
    x = torch.linspace(0.0, 1.0, n_elements * n_per_element + 1)
    pwi = PointWiseInterpolator(problem["mesh"], problem["sf"], problem["u"], problem["mapping"])
    w = pwi.at_position(x.unsqueeze(-1)).squeeze(-1).detach()
    return x, w


def rescale(x, w, u_axial, P, *, E, A, I, length):
    """Convert non-dimensional quantities back to physical ones.

    x_hat = l x, w_hat = r w, u_hat = (r^2 / l) u, P_hat = P E I / l^2,
    with r = sqrt(I / A).
    """
    r = (I / A) ** 0.5
    return (
        length * x,
        r * w,
        (r**2 / length) * u_axial,
        P * E * I / length**2,
    )
