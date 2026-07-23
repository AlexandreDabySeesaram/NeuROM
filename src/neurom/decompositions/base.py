from abc import ABC, abstractmethod

import torch.nn as nn


class TensorDecomposition(nn.Module, ABC):
    """A tensor/separated decomposition that can populate a FieldLayout.

    Concrete formats (CP, and later Tucker / TT) own their own factor fields and
    know how many they are. The contract :class:`neurom.neurom_model.NeuROMModel`
    relies on is: register those fields once (:meth:`register_into`), then at
    inference evaluate the field pointwise (:meth:`evaluate`) or assemble the
    full grid (:meth:`assemble`). Per-forward interpolation is the injected
    IntegrationDomain's job, not the decomposition's. Format-specific structure
    readback and rank/mode enrichment stay on the concrete subclass.
    """

    @abstractmethod
    def register_into(self, field_layout) -> None:
        """Register this decomposition's factor fields in the layout (setup)."""

    @abstractmethod
    def evaluate(self, coords):
        """Evaluate the field at matched query points (inference). Returns ``(P, d)``."""

    @abstractmethod
    def assemble(self, coords):
        """Assemble the full grid tensor over the given per-axis coordinates."""
