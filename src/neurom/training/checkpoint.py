"""Saving and reloading a trained model, so plotting need not retrain.

Training the 5-parametric beam takes minutes; changing a figure takes seconds.
These two functions are what separates the two.

What is saved is the model's ``state_dict`` plus the training history. That is
enough because of how the decomposition stores its truncation: ``active`` is a
registered **buffer** on every ``QuadratureAssembly``
(``neurom.interpolation.quadrature_assembly``), so how many modes were active
round-trips with the values themselves -- no separate bookkeeping, and no way
for the two to disagree.

What is deliberately **not** saved is the geometry: meshes, shape functions,
quadrature rules, the energy. A checkpoint is meaningless without the code that
built the model, so :func:`load_checkpoint` takes a freshly built model and
fills it, rather than pretending to reconstruct one. The ``metadata`` dict is
where a caller records what that model was (mesh sizes, strategy, criteria) so a
mismatch can be caught rather than silently plotted.
"""

import torch

FORMAT = 1


def save_checkpoint(path, model, history=None, metadata=None):
    """Write ``model``'s parameters and the run's history to ``path``.

    Args:
        path (str or Path): Destination file. Overwritten if it exists.
        model (torch.nn.Module): The trained model.
        history (TrainingHistory, optional): The run that produced it. Stored
            as-is; it is plain data (see :mod:`neurom.training.history`), no
            torch objects inside.
        metadata (dict, optional): Anything needed to tell later whether this
            checkpoint matches the model you are about to load it into --
            mesh sizes, strategy name, criteria. Never read by this module.

    Returns:
        Path: ``path``, for chaining.
    """
    torch.save(
        {
            "format": FORMAT,
            "state_dict": model.state_dict(),
            "history": history,
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def load_checkpoint(path, model, strict=True):
    """Fill ``model`` from the checkpoint at ``path``.

    ``model`` must be built exactly as it was when saved -- same axes, same mesh
    sizes, same ``n_modes_max``. A mismatch surfaces as a shape error from
    ``load_state_dict``, which is the intended behaviour: silently loading half
    a decomposition would produce plots that look plausible and are wrong.

    The freeze state (``requires_grad``) is **not** part of a ``state_dict`` and
    is therefore not restored. It does not need to be: a trainer's
    ``prepare_stage`` sets the whole freeze state at the start of every stage,
    so resuming training from a checkpoint is well defined regardless of how the
    parameters arrive. What *is* restored is which modes are active, since that
    lives in the ``active`` buffers.

    Args:
        path (str or Path): The checkpoint to read.
        model (torch.nn.Module): A freshly built model to fill, in place.
        strict (bool): Passed to ``load_state_dict``.

    Returns:
        tuple: ``(history, metadata)`` as saved -- ``history`` may be None if
        the checkpoint was written without one.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the checkpoint was written by a different format version.
    """
    # weights_only=False: the payload carries the TrainingHistory dataclass, not
    # only tensors. The checkpoints this reads are ones we wrote ourselves --
    # do not point it at a file from anywhere else.
    checkpoint = torch.load(path, weights_only=False)

    version = checkpoint.get("format")
    if version != FORMAT:
        raise ValueError(
            f"{path}: checkpoint format {version!r}, expected {FORMAT!r}. "
            "Retrain and save it again."
        )

    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return checkpoint.get("history"), checkpoint.get("metadata", {})
