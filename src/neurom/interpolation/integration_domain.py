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

    Static and dynamic assemblies
        The domain **owns** the assemblies whose existence is fixed for the
        whole run -- the displacement field of a FEM problem, a load, a source.
        Those are the ones passed to ``__init__``.

        A separated representation, on the other hand, grows monoms as it is
        enriched, so its assemblies cannot be stored here without going stale.
        They are passed *per call* instead, through the optional ``assemblies``
        argument of :meth:`interpolate_all` and :meth:`update_contexts`: the
        owner of the decomposition (:class:`neurom.neurom_model.NeuROMModel`)
        asks it for its current assemblies on every forward. Nothing to keep in
        sync -- what exists is interpolated.

    Args:
        assemblies (list[QuadratureAssembly]): The static quadrature assemblies
            of this integration domain.

    Attributes:
        assemblies (nn.ModuleList): The registered list of static
            ``QuadratureAssembly`` instances.
        _contexts (list[QuadratureContext]): Deduplicated list of unique
            ``QuadratureContext`` objects referenced by the *static*
            assemblies. Contexts reached only through per-call assemblies are
            deduplicated on the fly instead.
    """

    def __init__(self, assemblies: list[QuadratureAssembly]):
        super().__init__()
        self.assemblies = nn.ModuleList(assemblies)
        # deduplicated contexts — each unique context appears only once
        self._contexts: list[QuadratureContext] = list(
            {id(a.context): a.context for a in assemblies}.values()
        )

    def _all(self, assemblies=None) -> list[QuadratureAssembly]:
        """Static assemblies, followed by the per-call ones if any."""
        if assemblies is None:
            return list(self.assemblies)
        return [*self.assemblies, *assemblies]

    def update_contexts(self, assemblies: list[QuadratureAssembly] = None):
        """Recompute geometry for all unique contexts.

        Calls ``update()`` on every unique ``QuadratureContext`` reached from
        the static assemblies and from ``assemblies``. Should be called at the
        start of each forward pass when the mesh nodes are trainable and may
        have changed.

        Args:
            assemblies (list[QuadratureAssembly], optional): Per-call
                assemblies to update on top of the static ones -- typically
                ``decomposition.assemblies()``. Pass them whenever their
                geometry is trainable, otherwise their contexts are never
                refreshed: the cached ``_contexts`` covers the static
                assemblies only.
        """
        if assemblies is None:
            contexts = self._contexts
        else:
            contexts = {
                id(a.context): a.context for a in self._all(assemblies)
            }.values()
        for ctx in contexts:
            ctx.update()

    def interpolate_all(
        self, field_layout: "FieldLayout", assemblies: list[QuadratureAssembly] = None
    ):
        """Interpolate all fields and store the results in a ``FieldLayout``.

        Iterates over the static assemblies then over ``assemblies``, calls
        each one's ``interpolate()`` method, and stores the resulting
        ``QuadratureAssemblyResult`` in ``field_layout`` under the assembly's
        associated field.

        Args:
            field_layout (FieldLayout): The layout in which to record each
                interpolation result via ``field_layout.update()``.
            assemblies (list[QuadratureAssembly], optional): Per-call
                assemblies to interpolate on top of the static ones -- for a
                separated representation, ``decomposition.assemblies()``, whose
                length grows with every mode added.
        """
        # Interpolate all required fields and update() their values in FieldLayout
        for assembly in self._all(assemblies):
            field_layout.update(assembly.field, assembly.interpolate())
