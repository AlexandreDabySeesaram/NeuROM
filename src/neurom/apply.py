from dataclasses import replace
from typing import Callable
import torch

from neurom.samplings import Sampling


def batch(fn: Callable, batch_ndim: int) -> Callable:
    """Wrap a pure pointwise function with vmap to automatically batch over the first batch_ndim dimensions.

    Args:
        fn (Callable) : A pure pointwise function that operates on non-batched inputs.
        batch_ndim (int) : The number of leading dimensions to batch over.
    Returns:
        A new function that applies fn to inputs with batch_ndim leading dimensions, automatically vectorizing
    """
    f = fn
    for _ in range(batch_ndim):
        f = torch.vmap(f)
    return f


def apply(
    fn: Callable, *args: Sampling | torch.Tensor, **kwargs
) -> Sampling | torch.Tensor:
    """Apply a pure pointwise function to Sampling or Tensor arguments.

    Automatically batches over the ``batch_shape`` of the first Sampling argument.
    Preserves Sampling type via dataclasses.replace.

    Args:
        fn (Callable) : A pure pointwise function that operates on non-batched inputs.
        *args (Sampling | Tensor) : Arguments to apply the function to.
            If any argument is a Sampling, all Sampling arguments must have the same type and batch_shape.
        **kwargs : Keyword arguments to pass to the function, not batched.
    Returns:
        The result of applying the function, with the same type and batch_shape as the first Sampling
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
