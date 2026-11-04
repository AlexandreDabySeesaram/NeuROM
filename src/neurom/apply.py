"""Apply pure point-wise functions over batched ``Sampling`` fields via ``torch.vmap``."""

from dataclasses import replace
from typing import Callable
import torch

from neurom.samplings import Sampling


def batch(fn: Callable, batch_ndim: int) -> Callable:
    """Wrap a pure pointwise function with vmap to batch over the first ``batch_ndim`` dimensions.

    Applies ``torch.vmap`` repeatedly so that the returned function automatically
    vectorises over ``batch_ndim`` leading dimensions.

    Args:
        fn (Callable): A pure pointwise function that operates on non-batched inputs.
        batch_ndim (int): The number of leading dimensions to batch over.

    Returns:
        Callable: A new function that applies ``fn`` to inputs with ``batch_ndim``
        leading dimensions, automatically vectorising via nested ``torch.vmap``.
    """
    f = fn
    for _ in range(batch_ndim):
        f = torch.vmap(f)
    return f


def apply(
    fn: Callable, *args: Sampling | torch.Tensor, **kwargs
) -> Sampling | torch.Tensor:
    """Apply a pure pointwise function to ``Sampling`` or ``torch.Tensor`` arguments.

    Automatically batches ``fn`` over the ``batch_shape`` of the first
    ``Sampling`` argument using :func:`batch`.  The result is wrapped back into
    the same ``Sampling`` subclass via ``dataclasses.replace``.

    If no argument is a ``Sampling`` instance, ``fn`` is called directly on the
    provided arguments without any batching.

    Args:
        fn (Callable): A pure pointwise function that operates on non-batched inputs.
        args (Sampling | torch.Tensor): Positional arguments for ``fn``.
            If any argument is a ``Sampling``, all ``Sampling`` arguments must
            share the same concrete subtype and the same ``batch_shape``.
        kwargs: Keyword arguments forwarded to ``fn`` without batching.

    Returns:
        Sampling | torch.Tensor: The result of applying ``fn``.  When at least
        one argument is a ``Sampling``, the output is a new ``Sampling`` of the
        same subtype and ``batch_shape`` as the first ``Sampling`` argument.
        When no argument is a ``Sampling``, the raw return value of ``fn`` is
        returned unchanged.

    Raises:
        AssertionError: If two ``Sampling`` arguments have different subtypes or
            different ``batch_shape`` values.
    """
    ref = next((a for a in args if isinstance(a, Sampling)), None)
    if ref is None:
        return fn(*args, **kwargs)

    # validate consistency
    for a in args:
        if isinstance(a, Sampling):
            assert type(a) is type(ref), (
                f"apply() requires same Sampling type, got {type(a)} and {type(ref)}"
            )
            assert a.batch_shape == ref.batch_shape, (
                f"apply() requires same batch_shape, got {a.batch_shape} vs {ref.batch_shape}"
            )

    unwrapped = tuple(a.values if isinstance(a, Sampling) else a for a in args)
    result_values = batch(fn, batch_ndim=len(ref.batch_shape))(*unwrapped, **kwargs)
    return replace(ref, values=result_values)
