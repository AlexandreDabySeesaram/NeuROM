import torch


def jacobian(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute the Jacobian of u w.r.t. x using autograd.

    Args:
        x (torch.Tensor) : The variable with respect to which gradient is computed. Shape is (N_e, N_q, x_dim).
        u (torch.Tensor) : The field whose Jacobian we want. Shape is (N_e, N_q, *u_shape).

    Returns:
        torch.Tensor: The Jacobian of u w.r.t. x. Shape is (N_e, N_q, *u_shape, x_dim).
    """
    N_e, N_q, d = x.shape
    u_shape = u.shape[2:]
    m = int(torch.tensor(u_shape).prod().item()) if u_shape else 1

    u_flat = u.reshape(N_e, N_q, m)  # flatten field dims

    grads = []  # will collect d columns of the Jacobian
    for k in range(m):
        # grad of k-th output component w.r.t. all x
        g = torch.autograd.grad(
            u_flat[..., k].sum(),  # scalar
            x,
            retain_graph=True,
            create_graph=True,
        )[0]  # (N_e, N_q, d)
        grads.append(g)

    # stack into (N_e, N_q, m, d) then reshape to (N_e, N_q, *u_shape, d)
    J = torch.stack(grads, dim=2)  # (N_e, N_q, m, d)
    return J.reshape(N_e, N_q, *u_shape, d)
