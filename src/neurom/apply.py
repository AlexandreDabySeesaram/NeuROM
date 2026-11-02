"""Apply operations over batched ``Sampling`` fields, agnostic to which axes carry the batch.

Two engines, both ``Sampling``-native and batch-rank-agnostic. The richest ``Sampling`` argument (the
longest ``batch_shape``) defines the result subtype and batch shape; every other ``Sampling``'s
``batch_shape`` must be a **prefix** of it, so lower-rank operands broadcast over the trailing axes
(element data over quadrature/points/modes, etc.). Non-``Sampling`` positional args are constants.

- :func:`apply` -- **fast / default (no vmap).** Broadcast-aligns the operands to a common batch rank
  (zero-copy ``unsqueeze`` views) and calls ``fn`` once on the full-batch tensors. ``fn`` is written
  with trailing-axis ops / ``"..."``-einsum, which torch vectorises over any number of leading dims.
  Use for ops expressible as batched tensor ops (contractions, elementwise, batched ``torch.linalg``):
  the mapping, interpolation, most physics.
- :func:`apply_pointwise` -- **general (vmap).** Maps ``fn`` over the batch shape; ``fn`` operates on a
  single ``f_shape`` "point". Use for genuinely point-wise nonlinear ops that broadcast cannot express
  (per-point iterative solvers, per-point autodiff tangents via ``vmap(jacrev(...))``). Slower per call.
"""

from dataclasses import replace
from typing import Callable
import torch

from neurom.samplings import Sampling


def _richest(samplings: list[Sampling]) -> Sampling:
    """Return the richest ``Sampling`` (longest ``batch_shape``); validate the prefix rule.

    Args:
        samplings (list[Sampling]): The ``Sampling`` arguments (non-empty).

    Returns:
        Sampling: The argument with the longest ``batch_shape``.

    Raises:
        ValueError: If any ``batch_shape`` is not a prefix of the richest one.
    """
    ref = max(samplings, key=lambda s: len(s.batch_shape))
    for s in samplings:
        r = len(s.batch_shape)
        if s.batch_shape != ref.batch_shape[:r]:
            raise ValueError(
                f"apply(): batch_shape {tuple(s.batch_shape)} is not a prefix of the common "
                f"batch_shape {tuple(ref.batch_shape)}; operands must share their leading axes."
            )
    return ref


def apply(fn: Callable, *args: Sampling | torch.Tensor, **kwargs) -> Sampling | torch.Tensor:
    """Fast engine: broadcast-align the operands and call ``fn`` once on the full-batch tensors.

    Each lower-rank ``Sampling`` has singleton axes inserted after its own batch axes (zero-copy views)
    so a trailing-``...`` op in ``fn`` broadcasts it over the extra axes. No ``vmap``.

    Args:
        fn (Callable): Operates on the aligned full-batch tensors (write it with trailing-axis ops /
            ``"..."``-einsum). Receives one tensor per positional arg, in order.
        args (Sampling | torch.Tensor): Positional operands; ``Sampling`` batch shapes must be
            prefix-compatible. Non-``Sampling`` args are passed through unchanged (constants).
        kwargs: Forwarded to ``fn`` unchanged.

    Returns:
        Sampling | torch.Tensor: ``fn``'s result wrapped in the richest ``Sampling``'s subtype/batch,
        or the raw result if no argument is a ``Sampling``.
    """
    samplings = [a for a in args if isinstance(a, Sampling)]
    if not samplings:
        return fn(*args, **kwargs)
    ref = _richest(samplings)
    rank = len(ref.batch_shape)
    aligned = []
    for a in args:
        if not isinstance(a, Sampling):
            aligned.append(a)
            continue
        v, r = a.values, len(a.batch_shape)
        for _ in range(rank - r):
            v = v.unsqueeze(r)  # singleton axis after this operand's own batch axes
        aligned.append(v)
    return replace(ref, values=fn(*aligned, **kwargs))


def _nested_vmap(fn: Callable, ranks: tuple[int, ...], n_axes: int) -> Callable:
    """Wrap ``fn`` in ``n_axes`` nested ``vmap``s, broadcasting args that lack a given batch axis.

    For original batch axis ``l`` (``0`` = outermost) an argument is mapped (``in_dims=0``) iff
    ``l < ranks[i]`` and broadcast (``in_dims=None``) otherwise.

    Args:
        fn (Callable): Point-wise function over single ``f_shape`` tensors.
        ranks (tuple[int, ...]): Per-argument batch rank (``0`` = constant).
        n_axes (int): Common batch rank.

    Returns:
        Callable: ``fn`` vectorised over ``n_axes`` batch axes with per-axis broadcasting.
    """
    f = fn
    for axis in range(n_axes - 1, -1, -1):
        in_dims = tuple(0 if axis < r else None for r in ranks)
        f = torch.vmap(f, in_dims=in_dims)
    return f


def apply_pointwise(
    fn: Callable, *args: Sampling | torch.Tensor, **kwargs
) -> Sampling | torch.Tensor:
    """General engine: ``vmap`` ``fn`` over the batch shape; ``fn`` operates on a single point.

    For ops that cannot be written as batched tensor ops -- per-point iterative solvers, per-point
    autodiff tangents (compose with ``torch.func.jacrev``), data-dependent per-point logic.

    Args:
        fn (Callable): Operates on single ``f_shape`` tensors (one "point").
        args (Sampling | torch.Tensor): Positional operands; ``Sampling`` batch shapes must be
            prefix-compatible. Non-``Sampling`` args are constants (broadcast over every batch axis).
        kwargs: Forwarded to ``fn`` unchanged (unbatched).

    Returns:
        Sampling | torch.Tensor: ``fn``'s result wrapped in the richest ``Sampling``'s subtype/batch,
        or the raw result if no argument is a ``Sampling``.
    """
    samplings = [a for a in args if isinstance(a, Sampling)]
    if not samplings:
        return fn(*args, **kwargs)
    ref = _richest(samplings)
    ranks = tuple(len(a.batch_shape) if isinstance(a, Sampling) else 0 for a in args)
    unwrapped = tuple(a.values if isinstance(a, Sampling) else a for a in args)
    result = _nested_vmap(fn, ranks, len(ref.batch_shape))(*unwrapped, **kwargs)
    return replace(ref, values=result)
