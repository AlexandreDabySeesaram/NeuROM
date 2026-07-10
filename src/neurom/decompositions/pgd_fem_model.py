import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class PGDFEMModel(nn.Module):
    """FEM model driving a tensor decomposition through a FieldLayout.

    PGD analogue of :class:`neurom.fem_model.FEMModel`. Depends only on the
    :class:`TensorDecomposition` contract, so any format (CP, and later Tucker /
    TT) drives the same model. Registers the decomposition's factor fields into
    the layout at construction; each forward re-interpolates the active ones and
    evaluates the (external) loss.

    Args:
        decomposition (TensorDecomposition): The separated representation.
        field_layout (FieldLayout): Flat layout the decomposition fills.
        loss (Callable[[], torch.Tensor]): No-arg callable returning the scalar
            energy, closed over the decomposition, layout and problem data.
    """

    def __init__(self, decomposition: TensorDecomposition, field_layout, loss):
        super().__init__()
        self.decomposition = decomposition
        self.field_layout = field_layout
        self.loss = loss
        decomposition.register_into(field_layout)

    def forward(self):
        self.decomposition.fill(self.field_layout)
        return self.loss()
