from functools import singledispatch

import torch

from neurom.samplings import Sampling


def _jacobian_flat_impl(x: torch.Tensor, u_flat: torch.Tensor) -> torch.Tensor:
    """Compute jacobian on flat tensors

    Implementation detail.
    """
    m = u_flat.shape[-1]
    grads = []
    for k in range(m):
        g = torch.autograd.grad(
            u_flat[..., k].sum(),
            x,
            retain_graph=True,
            create_graph=True,
        )[0]  # (*batch_shape, x_dim)
        grads.append(g)

    # (*batch_shape, m, x_dim) -> (*batch_shape, *f_shape, x_dim)
    J_flat = torch.stack(grads, dim=-2)  # (*batch_shape, m, x_dim)
    return J_flat


@singledispatch
def jacobian(x, u):
    raise TypeError(f"Unsupported types: {type(x)}, {type(u)}")


@jacobian.register
def _(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute jacobian of a function

    Jacobian of  ``u`` w.r.t. ``x`` via autograd on precomputed tensors.

    Args:
        x (Sampling | torch.Tensor): (*batch_shape, x_dim). Needs to have requires_grad=True
        u(Sampling | torch.Tensor): (*batch_shape, *f_shape). Function of x via graph
    Returns:
        torch.Tensor with computed jacobian of shape (*batch_shape, *f_shape, x_dim) or Sampling with same shape.
    """
    # infer batch_shape and f_shape from x and u
    # x has one more trailing dim (x_dim) than batch_shape
    batch_ndim = x.ndim - 1
    batch_shape = x.shape[:batch_ndim]  # (*batch_shape,)
    x_dim = x.shape[-1]
    f_shape = u.shape[batch_ndim:]  # (*f_shape,) — may be () (2,) (2,2) etc.
    m = f_shape.numel() if len(f_shape) > 0 else 1

    # flatten f_shape into a single dim for uniform treatment
    u_flat = u.reshape(*batch_shape, m)  # (*batch_shape, m)

    J_flat = _jacobian_flat_impl(x, u_flat)
    return J_flat.reshape(*batch_shape, *f_shape, x_dim)


@jacobian.register
def _(x: Sampling, u: Sampling) -> Sampling:
    """Compute jacobian of a function

    Jacobian of  ``u`` w.r.t. ``x`` via autograd on precomputed tensors.

    Args:
        x (Sampling | torch.Tensor): (*batch_shape, x_dim). Needs to have requires_grad=True
        u(Sampling | torch.Tensor): (*batch_shape, *f_shape). Function of x via graph
    Returns:
        torch.Tensor with computed jacobian of shape (*batch_shape, *f_shape, x_dim) or Sampling with same shape.
    """
    assert type(x) == type(u), (
        f"x and u must be of the same Sampling type but got x of type '{type(x)}' and u of type '{type(u)}'"
    )

    assert x.batch_shape == u.batch_shape, (
        f"batch shapes don't match: x.batch_shape={x.batch_shape}, u.batch_shape={u.batch_shape}"
    )

    batch_shape = x.batch_shape
    x_dim = x.f_shape[0]
    f_shape = u.f_shape
    m = f_shape.numel() if len(f_shape) > 0 else 1

    # flatten f_shape into a single dim for uniform treatment
    u_flat = u.values.reshape(*batch_shape, m)  # (*batch_shape, m)

    J_flat = _jacobian_flat_impl(x.values, u_flat)

    return u.__class__(values=J_flat.reshape(*batch_shape, *f_shape, x_dim))
