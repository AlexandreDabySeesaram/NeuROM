from abc import ABC, abstractmethod

import torch.nn as nn


class TensorDecomposition(nn.Module, ABC):
    """A tensor/separated decomposition that can populate a FieldLayout.

    Concrete formats (CP, and later Tucker / TT) own their own factor fields and
    know how many they are. The only contract :class:`PGDFEMModel` relies on is:
    register those fields once (:meth:`register_into`), then re-interpolate the
    active ones per forward (:meth:`fill`). Format-specific structure readback,
    assembly and rank/mode enrichment stay on the concrete subclass.
    """

    @abstractmethod
    def register_into(self, field_layout) -> None:
        """Register this decomposition's factor fields in the layout (setup)."""

    @abstractmethod
    def fill(self, field_layout) -> None:
        """Interpolate the active factor fields and ``update()`` them in the layout.

        Called once per forward: the PGD analogue of
        :meth:`neurom.interpolation.integration_domain.IntegrationDomain.interpolate_all`.
        """
