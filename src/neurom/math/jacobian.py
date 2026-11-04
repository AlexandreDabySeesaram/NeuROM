"""Jacobian of a field with respect to its spatial coordinates via autograd."""

from functools import singledispatch

import torch

from neurom.samplings import Sampling


def _jacobian_flat_impl(x: torch.Tensor, u_flat: torch.Tensor) -> torch.Tensor:
    """Compute the Jacobian of a flattened output tensor with respect to inputs.

    Internal implementation detail used by :func:`jacobian`.  Iterates over
    the ``m`` output components of the flattened field ``u_flat`` and collects
    their gradients with respect to ``x`` via ``torch.autograd.grad``.

    Args:
        x (torch.Tensor): Input tensor of shape ``(*batch_shape, x_dim)`` with
            ``requires_grad=True``.
        u_flat (torch.Tensor): Flattened output tensor of shape
            ``(*batch_shape, m)`` that depends on ``x`` through the autograd
            graph.

    Returns:
        torch.Tensor: Jacobian tensor of shape ``(*batch_shape, m, x_dim)``.
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
    """Compute the Jacobian of ``u`` with respect to ``x``.

    Dispatches to the appropriate implementation based on the types of ``x``
    and ``u``.  Supported type pairs are ``(torch.Tensor, torch.Tensor)`` and
    ``(Sampling, Sampling)``.

    Args:
        x: Input coordinates.  Must be ``torch.Tensor`` or a ``Sampling``
            subclass with ``requires_grad=True`` on the underlying values.
        u: Output field that depends on ``x`` through the autograd graph.
            Must be the same type as ``x``.

    Returns:
        The Jacobian with the same type as the inputs, with shape
        ``(*batch_shape, *f_shape, x_dim)``.

    Raises:
        TypeError: If the types of ``x`` and ``u`` are not supported.
    """
    raise TypeError(f"Unsupported types: {type(x)}, {type(u)}")


@jacobian.register
def _(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute the Jacobian of a tensor field with respect to input coordinates.

    Computes the Jacobian of ``u`` with respect to ``x`` via autograd on
    precomputed tensors.  The field shape ``f_shape`` is inferred from ``u``
    and ``x``.

    Args:
        x (torch.Tensor): Input coordinates of shape ``(*batch_shape, x_dim)``
            with ``requires_grad=True``.
        u (torch.Tensor): Output field of shape ``(*batch_shape, *f_shape)``
            that depends on ``x`` through the autograd graph.

    Returns:
        torch.Tensor: Jacobian tensor of shape
        ``(*batch_shape, *f_shape, x_dim)``.
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
    """Compute the Jacobian of a Sampling field with respect to input coordinates.

    Computes the Jacobian of ``u`` with respect to ``x`` via autograd on
    precomputed tensors.  Both arguments must be the same ``Sampling`` subclass
    and share the same ``batch_shape``.

    Args:
        x (Sampling): Input coordinates Sampling of field shape ``(x_dim,)``
            whose ``values`` have ``requires_grad=True``.
        u (Sampling): Output field Sampling of field shape ``(*f_shape)``
            that depends on ``x`` through the autograd graph.  Must be the
            same type and ``batch_shape`` as ``x``.

    Returns:
        Sampling: A Sampling of the same type as ``u`` whose ``values`` have
        shape ``(*batch_shape, *f_shape, x_dim)``.
    """
    assert type(x) is type(u), (
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
