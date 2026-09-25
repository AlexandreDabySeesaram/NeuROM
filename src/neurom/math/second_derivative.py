"""Second derivative of a field with respect to its spatial coordinates via autograd."""

import torch

from neurom.math.jacobian import jacobian
from neurom.samplings import Sampling


def second_derivative(x, u):
    """Compute the second derivative of ``u`` with respect to ``x``.

    Applies :func:`jacobian` twice; the graph is kept so the result stays
    differentiable.  If the first derivative no longer depends on ``x``
    (``u`` affine in ``x``), zeros of the right shape are returned.

    Args:
        x: Input coordinates (``torch.Tensor`` or ``Sampling``) with
            ``requires_grad=True``.
        u: Field that depends on ``x`` through the autograd graph.

    Returns:
        The second derivative, of shape ``(*batch_shape, *f_shape, x_dim, x_dim)``.
    """
    du = jacobian(x, u)
    if isinstance(du, Sampling):
        if not du.values.requires_grad:
            x_dim = x.f_shape[0]
            return du.__class__(values=torch.zeros(*du.values.shape, x_dim, dtype=du.values.dtype, device=du.values.device))
    elif not du.requires_grad:
        return torch.zeros(*du.shape, x.shape[-1], dtype=du.dtype, device=du.device)
    return jacobian(x, du)
