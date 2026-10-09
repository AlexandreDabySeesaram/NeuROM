"""Per-element field values (output / export), e.g. projected stresses."""

import torch
import torch.nn as nn

from neurom.samplings import ElementSampling


class ElementField(nn.Module):
    """Per-element output values, one vector per element, shape ``(n_elem, f_dim)``.

    This is an **output / export holder** for quantities evaluated per element
    (material properties, phase indicators, projected stresses written as XDMF
    cell data) -- not an interpolatable finite-element space.

    A *trainable* piecewise-constant (DG0) unknown is instead a field on a
    ``FunctionSpace`` built from the :func:`neurom.elements.DG0` element -- it needs
    no dedicated field type.

    Attributes:
        name (str): Human-readable identifier for the field.
        values (torch.Tensor or torch.nn.Parameter): Per-element values of shape
            ``(n_elem, f_dim)``.  An ``nn.Parameter`` when ``trainable=True``,
            else a registered buffer.
    """

    def __init__(self, name: str, values: torch.Tensor, trainable: bool = False):
        """Initialize an ElementField.

        Args:
            name (str): Human-readable identifier for the field.
            values (torch.Tensor): Per-element field values of shape
                ``(n_elem, f_dim)``.
            trainable (bool): If ``True``, ``values`` is registered as an
                ``nn.Parameter``; otherwise as a non-trainable buffer.
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
