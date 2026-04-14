import torch
import torch.nn as nn


class ElementField(nn.Module):
    """
    Field whose values are defined per element (not per node).
    Typical use: material properties, phase indicators, per-element outputs.

    Args:
        values: (n_elem, f_dim)
    """

    def __init__(self, name: str, values: torch.Tensor, trainable: bool = False):
        super().__init__()
        self.name = name
        if trainable:
            self.values = nn.Parameter(values)
        else:
            self.register_buffer("values", values)
