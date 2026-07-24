"""Cheap degeneracy diagnostics on a CP decomposition.

Every mode is seeded with the same ``Axis.init_values``, so nothing in the
optimisation *guarantees* that two modes end up different. Whatever keeps them
apart -- freezing, in the greedy strategy -- is a hope until it is measured, so
these helpers measure it.

They live here rather than on a trainer because more than one strategy needs
them and they only ever read the decomposition: no forward pass, no model, no
optimizer.

Caveat shared by all of them: they use raw nodal vectors, not the
quadrature-weighted L2 inner product, so this is *not* the energy-norm
correlation. Cheap and good enough to spot duplication, not to quantify it.
"""

import math


def monom_values(decomposition, mode):
    """Full nodal values of every monom of ``mode`` (constrained DOFs included).

    ``full_values()`` is correct here because the space axis's Dirichlet values
    are homogeneous (zero): every mode's constrained DOFs are zero too, so
    including them does not bias the correlation. With **inhomogeneous**
    Dirichlet data every mode would share the same nonzero constant component on
    those DOFs, and the correlations below would be biased upward regardless of
    how different the free DOFs are.

    Args:
        decomposition (CPPGD): The decomposition to read.
        mode (int): Index of the mode whose monoms to read.

    Returns:
        list[torch.Tensor]: One detached tensor per monom/axis of ``mode``.
    """
    return [field.full_values().detach() for field in decomposition.monoms[mode]]


def amplitude(values):
    """The mode's overall size, ``prod_k ||w^k||``.

    Args:
        values (list[torch.Tensor]): A mode's monoms, as from :func:`monom_values`.

    Returns:
        float: The product of the per-axis norms.
    """
    product = 1.0
    for tensor in values:
        product *= float(tensor.norm())
    return product


def correlation(first, second):
    """Normalised inner product of two rank-1 tensors, in absolute value.

    ``|prod_k <w_i^k, w_j^k> / (||w_i^k|| ||w_j^k||)|`` -- 1 when the two modes
    are parallel on every axis, i.e. copies of each other up to the CP scale
    invariance.

    A zero-norm axis makes the quotient undefined; it is reported as 0.0, since
    a mode that vanishes on one axis is the zero tensor and duplicates nothing.
    NaN (from a diverged stage) propagates rather than being swallowed.

    Args:
        first, second (list[torch.Tensor]): Two modes' monoms, axis-aligned.

    Returns:
        float: The correlation in [0, 1], or NaN if either mode is not finite.
    """
    product = 1.0
    for a, b in zip(first, second):
        norms = a.norm() * b.norm()
        if norms == 0.0:
            return 0.0
        product *= float((a * b).sum() / norms)
    return abs(product)


def _largest(values):
    """Largest of ``values``, propagating NaN instead of hiding it.

    ``max(0.0, nan)`` returns ``0.0`` in Python -- ``max`` keeps its first
    argument when a comparison is False, and every comparison with NaN is --
    which would report perfect orthogonality exactly when the state is garbage,
    the opposite of what the diagnostic is for.

    Args:
        values (Iterable[float]): The correlations to reduce.

    Returns:
        float: The maximum, 0.0 if empty, NaN if any value is NaN.
    """
    largest = 0.0
    for value in values:
        if math.isnan(value) or math.isnan(largest):
            largest = math.nan
        else:
            largest = max(largest, value)
    return largest


def max_correlation(decomposition, mode):
    """How much ``mode`` duplicates a mode that precedes it.

    The greedy question: a tiny gain together with a value near 1 is the
    sequence rediscovering a mode it already has.

    Args:
        decomposition (CPPGD): The decomposition to read.
        mode (int): Index of the mode to test against its predecessors.

    Returns:
        float: The largest correlation against modes ``0..mode-1``; 0.0 for
            ``mode == 0``.
    """
    current = monom_values(decomposition, mode)
    return _largest(
        correlation(current, monom_values(decomposition, earlier))
        for earlier in range(mode)
    )


def max_pairwise_correlation(decomposition, n_modes=None):
    """The largest correlation over **every** pair of active modes.

    What :func:`max_correlation` cannot see, and what strategies that keep old
    modes trainable need: when all modes move at once, two *earlier* modes can
    collapse onto each other long after either was added.

    Args:
        decomposition (CPPGD): The decomposition to read.
        n_modes (int, optional): How many leading modes to consider. Defaults to
            the active ones.

    Returns:
        float: The largest pairwise correlation; 0.0 when there is at most one
            mode.
    """
    if n_modes is None:
        n_modes = decomposition.n_modes_truncated
    values = [monom_values(decomposition, m) for m in range(n_modes)]
    return _largest(
        correlation(values[i], values[j])
        for i in range(n_modes)
        for j in range(i + 1, n_modes)
    )
