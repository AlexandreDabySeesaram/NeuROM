import torch

from neurom.interpolation.integration_domain import IntegrationDomain


class SeparatedDomain(IntegrationDomain):
    """IntegrationDomain over mode-blocked assemblies; interpolates only active modes.

    Assemblies are grouped into blocks (one block per mode, each block one
    :class:`QuadratureAssembly` per axis). :meth:`interpolate_all` interpolates
    only the first ``n_active_modes`` blocks and ``update()``s them in the
    layout; :meth:`grow` activates the next block (greedy PGD enrichment). This
    is the truncation-aware analogue of :class:`IntegrationDomain`, which
    interpolates every assembly.

    Args:
        mode_blocks (list[list[QuadratureAssembly]]): One block per mode; each
            block holds one assembly per axis.
        n_active_modes (int): Number of initially active (interpolated) blocks.
    """

    def __init__(self, mode_blocks, n_active_modes):
        flat = [a for block in mode_blocks for a in block]
        super().__init__(flat)                     # dedups contexts, registers assemblies
        self._mode_blocks = mode_blocks            # same objects as self.assemblies
        self.register_buffer("n_active_modes", torch.tensor(int(n_active_modes)))

    def grow(self):
        """Activate the next mode-block. Returns its index. Raises at capacity."""
        if int(self.n_active_modes) >= len(self._mode_blocks):
            raise RuntimeError("Cannot grow: all mode-blocks already active.")
        idx = int(self.n_active_modes)
        self.n_active_modes += 1
        return idx

    def interpolate_all(self, field_layout):
        for block in self._mode_blocks[: int(self.n_active_modes)]:
            for assembly in block:
                field_layout.update(assembly.field, assembly.interpolate())
