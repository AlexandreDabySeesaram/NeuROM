import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class NeuROMModel(nn.Module):
    """Decomposition-driven model that reads like a classic ``nn.Module``.

    Counterpart of :class:`neurom.fem_model.FEMModel` for separated
    representations. Depends only on the :class:`TensorDecomposition` contract,
    so any format (CP, later Tucker/TT) drives the same model.

    ``forward`` branches on ``self.training``:
      * training: fill the layout via the decomposition and **return the filled
        ``field_layout``** — the intermediate output an external ``energy``
        consumes (``output = model(); loss = model.energy(output)``).
      * inference: ``forward(coords)`` returns the matched-pointwise field
        (``decomposition.evaluate(coords)``).

    Args:
        field_layout (FieldLayout): Fresh layout; ``__init__`` registers the
            decomposition's factor fields into it (a layout already holding those
            names raises ``ValueError`` on the duplicate registration).
        decomposition (TensorDecomposition): The separated representation.
        energy (Callable): Injected callable ``energy(output) -> torch.Tensor``
            (the counterpart of ``FEMModel.loss``), reading modes from the layout.
    """

    def __init__(self, field_layout, decomposition: TensorDecomposition, energy):
        super().__init__()
        self.field_layout = field_layout
        self.decomposition = decomposition
        self.energy = energy
        decomposition.register_into(field_layout)

    def forward(self, coords=None):
        if self.training:
            self.decomposition.fill(self.field_layout)
            return self.field_layout
        if coords is None:
            raise ValueError(
                "eval forward requires coords (one 1-D tensor per axis, matched length)."
            )
        return self.decomposition.evaluate(coords)

    def assemble(self, coords):
        return self.decomposition.assemble(coords)
