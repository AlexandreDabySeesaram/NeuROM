"""Field defined per mesh element rather than per node."""

import torch
import torch.nn as nn

from neurom.samplings import ElementSampling


class ElementField(nn.Module):
    """Field whose values are defined per element (not per node).

    Typical use-cases include material properties, phase indicators, and
    per-element output quantities.  The field stores one value vector per
    element, giving a values tensor of shape ``(n_elem, f_dim)``.

    Attributes:
        name (str): Human-readable identifier for the field.
        values (torch.Tensor or torch.nn.Parameter): Per-element field values
            of shape ``(n_elem, f_dim)``.  Stored as an ``nn.Parameter`` when
            ``trainable=True``, or as a registered buffer otherwise.
    """

    def __init__(self, name: str, values: torch.Tensor, trainable: bool = False):
        """Initialize an ElementField.

        Args:
            name (str): Human-readable identifier for the field.
            values (torch.Tensor): Per-element field values of shape
                ``(n_elem, f_dim)``.
            trainable (bool): If ``True``, ``values`` is registered as an
                ``nn.Parameter`` so gradients are computed during training.
                If ``False`` (default), ``values`` is registered as a
                non-trainable buffer.
        """
        super().__init__()
        self.name = name
        if trainable:
            self.values = nn.Parameter(values)
        else:
            self.register_buffer("values", values)

    def as_sampling(self) -> ElementSampling:
        """Return the field values wrapped in an ``ElementSampling`` object.

        Returns:
            ElementSampling: An ``ElementSampling`` whose ``values`` attribute
            equals ``self.values``.
        """
        return ElementSampling(values=self.values)
