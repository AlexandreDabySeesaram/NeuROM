"""Integration domain coordinating quadrature contexts and field interpolation."""

from typing import TYPE_CHECKING

import torch.nn as nn

from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.quadrature_context import QuadratureContext

if TYPE_CHECKING:
    from neurom.field_layout import FieldLayout


class IntegrationDomain(nn.Module):
    """Collection of ``QuadratureAssembly`` objects sharing deduplicated contexts.

    Groups multiple ``QuadratureAssembly`` instances that may share the same
    ``QuadratureContext`` (i.e. the same geometric setup). Duplicate contexts
    are stored only once so that geometry computations are not repeated.

    Args:
        assemblies (list[QuadratureAssembly]): The list of quadrature assemblies
            that make up this integration domain.

    Attributes:
        assemblies (nn.ModuleList): The registered list of
            ``QuadratureAssembly`` instances.
        _contexts (list[QuadratureContext]): Deduplicated list of unique
            ``QuadratureContext`` objects referenced by ``assemblies``.
    """

    def __init__(self, assemblies: list[QuadratureAssembly]):
        super().__init__()
        self.assemblies = nn.ModuleList(assemblies)
        # deduplicated contexts — each unique context appears only once
        self._contexts: list[QuadratureContext] = list(
            {id(a.context): a.context for a in assemblies}.values()
        )

    def update_contexts(self):
        """Recompute geometry for all unique contexts.

        Calls ``update()`` on every unique ``QuadratureContext`` held by this
        domain. Should be called at the start of each forward pass when the
        mesh nodes are trainable and may have changed.
        """
        for ctx in self._contexts:
            ctx.update()

    def interpolate_all(self, field_layout: "FieldLayout"):
        """Interpolate all fields and store the results in a ``FieldLayout``.

        Iterates over every ``QuadratureAssembly`` in ``self.assemblies``,
        calls its ``interpolate()`` method, and stores the resulting
        ``QuadratureAssemblyResult`` in ``field_layout`` under the assembly's
        associated field.

        Args:
            field_layout (FieldLayout): The layout in which to record each
                interpolation result via ``field_layout.update()``.
        """
        # Interpolate all required fields and update() their values in FieldLayout
        for assembly in self.assemblies:
            result = assembly.interpolate()
            field_layout.update(assembly.field, result)
