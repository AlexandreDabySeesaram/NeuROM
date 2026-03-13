import torch
from torch.func import jacrev, vmap


def jacobian_field(x, u):
    """
    x: (N_e, N_q, d)
    u: (N_e, N_q, *u_shape)
    returns: (N_e, N_q, *u_shape, d)
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
