import torch.nn as nn

from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.quadrature_assembly_result import QuadratureAssemblyResult
from neurom.interpolation.quadrature_context import QuadratureContext


class IntegrationDomain(nn.Module):
    """
    Collection of QuadratureAssemblies sharing possibly common QuadratureContexts.
    Contexts are deduplicated: multiple assemblies can share the same geometric setup.
    """

    def __init__(self, assemblies: list[QuadratureAssembly]):
        super().__init__()
        self.assemblies = nn.ModuleList(assemblies)
        # deduplicated contexts — each unique context appears only once
        self._contexts: list[QuadratureContext] = list(
            {id(a.context): a.context for a in assemblies}.values()
        )

    def update_contexts(self):
        """Recompute geometry for all unique contexts. Call at start of
        forward pass if mesh nodes are trainable."""
        for ctx in self._contexts:
            ctx.update()

    def interpolate_all(self, field_layout):
        # Interpolate every *active* field and update() its values in the layout.
        # Inactive assemblies (e.g. not-yet-enriched PGD modes) are skipped so no
        # autograd graph is built for them.
        for assembly in self.assemblies:
            if not bool(assembly.active):
                continue
            field_layout.update(assembly.field, assembly.interpolate())
