from abc import ABC, abstractmethod

import torch.nn as nn


class TensorDecomposition(nn.Module, ABC):
    """A tensor/separated decomposition that can populate a FieldLayout.

    Concrete formats (CP, and later Tucker / TT) own their own factor fields and
    know how many they are. The contract :class:`neurom.neurom_model.NeuROMModel`
    relies on is: keep the layout in phase with those fields
    (:meth:`register_into`), report the assemblies that interpolate them
    (:meth:`assemblies`), then at inference evaluate the field pointwise
    (:meth:`evaluate`) or assemble the full grid (:meth:`assemble`).

    A decomposition may grow while training -- enrichment appends fields and
    assemblies -- which is why the model re-reads :meth:`assemblies` on every
    forward instead of caching it, and why :meth:`register_into` must tolerate
    being called again. Performing the interpolation is the injected
    IntegrationDomain's job, not the decomposition's. Format-specific structure
    readback and rank/mode enrichment stay on the concrete subclass.
    """

    @abstractmethod
    def register_into(self, field_layout) -> None:
        """Bring the layout in phase with this decomposition's fields.

        Called at setup and again after any enrichment, so implementations must
        be idempotent.
        """

    @abstractmethod
    def assemblies(self) -> list:
        """The ``QuadratureAssembly`` interpolating this decomposition's fields.

        Re-read on every forward, since enrichment makes the list grow.
        """

    @abstractmethod
    def evaluate(self, coords):
        """Evaluate the field at matched query points (inference). Returns ``(P, d)``."""

    @abstractmethod
    def assemble(self, coords):
        """Assemble the full grid tensor over the given per-axis coordinates."""
