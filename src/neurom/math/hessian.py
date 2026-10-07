"""Hessian of a field with respect to its spatial coordinates via autograd."""

from functools import singledispatch

import torch

from neurom.math.jacobian import jacobian
from neurom.samplings import Sampling


@singledispatch
def hessian(x, u):
    """Compute the Hessian of ``u`` with respect to ``x``.

    Applies :func:`jacobian` twice; the graph is kept so the result stays
    differentiable.  Dispatches to the appropriate implementation based on the
    types of ``x`` and ``u``.  Supported type pairs are
    ``(torch.Tensor, torch.Tensor)`` and ``(Sampling, Sampling)``.

    Args:
        x: Input coordinates.  Must be ``torch.Tensor`` or a ``Sampling``
            subclass with ``requires_grad=True`` on the underlying values.
        u: Output field that depends on ``x`` through the autograd graph.
            Must be the same type as ``x``.

    Returns:
        The Hessian with the same type as the inputs, with shape
        ``(*batch_shape, *f_shape, x_dim, x_dim)``.

    Raises:
        TypeError: If the types of ``x`` and ``u`` are not supported.

    Note:
        The Hessian is usually defined for scalar fields only.  Here it is
        extended to vector and tensor fields component-wise:
        :math:`H_{ijk} = \\partial^2 u_i / \\partial x_j \\partial x_k`.
    """
    raise TypeError(f"Unsupported types: {type(x)}, {type(u)}")


@hessian.register
def _(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute the Hessian of a tensor field with respect to input coordinates.

    If the first derivative no longer depends on ``x`` (``u`` affine in
    ``x``), zeros of the right shape are returned: the Hessian is then
    genuinely zero.  This cannot hide a missing ``requires_grad`` on ``x`` or
    a ``u`` independent of ``x``: both make the first :func:`jacobian` call
    raise a ``RuntimeError``.

    Args:
        x (torch.Tensor): Input coordinates of shape ``(*batch_shape, x_dim)``
            with ``requires_grad=True``.
        u (torch.Tensor): Output field of shape ``(*batch_shape, *f_shape)``
            that depends on ``x`` through the autograd graph.

    Returns:
        torch.Tensor: Hessian tensor of shape
        ``(*batch_shape, *f_shape, x_dim, x_dim)``.
    """
    du = jacobian(x, u)
    # constant first derivative: autograd drops its graph, the Hessian is zero
    if not du.requires_grad:
        return torch.zeros(*du.shape, x.shape[-1], dtype=du.dtype, device=du.device)
    return jacobian(x, du)


@hessian.register
def _(x: Sampling, u: Sampling) -> Sampling:
    """Compute the Hessian of a Sampling field with respect to input coordinates.

    If the first derivative no longer depends on ``x`` (``u`` affine in
    ``x``), zeros of the right shape are returned: the Hessian is then
    genuinely zero.  This cannot hide a missing ``requires_grad`` on ``x`` or
    a ``u`` independent of ``x``: both make the first :func:`jacobian` call
    raise a ``RuntimeError``.

    Args:
        x (Sampling): Input coordinates Sampling of field shape ``(x_dim,)``
            whose ``values`` have ``requires_grad=True``.
        u (Sampling): Output field Sampling of field shape ``(*f_shape)``
            that depends on ``x`` through the autograd graph.  Must be the
            same type and ``batch_shape`` as ``x``.

    Returns:
        Sampling: A Sampling of the same type as ``u`` whose ``values`` have
        shape ``(*batch_shape, *f_shape, x_dim, x_dim)``.
    """
    du = jacobian(x, u)
    # constant first derivative: autograd drops its graph, the Hessian is zero
    if not du.values.requires_grad:
        x_dim = x.f_shape[0]
        return du.__class__(values=torch.zeros(*du.values.shape, x_dim, dtype=du.values.dtype, device=du.values.device))
    return jacobian(x, du)
