import torch.nn as nn

from neurom.decompositions.base import TensorDecomposition


class NeuROMModel(nn.Module):
    """Decomposition-driven model that reads like a classic ``nn.Module``.

    Counterpart of :class:`neurom.fem_model.FEMModel` for separated
    representations. Depends only on the :class:`TensorDecomposition` contract
    (``register_into`` / ``evaluate`` / ``assemble``), so any format (CP, later
    Tucker/TT) drives the same model. Interpolation is the injected
    :class:`~neurom.interpolation.integration_domain.IntegrationDomain`'s job —
    the same domain interpolates the decomposition's factor fields **and** any
    other field the energy reads (loads, sources).

    ``forward`` branches on ``self.training``:
      * training: interpolate every active field through the domain and **return
        the filled ``field_layout``** — the intermediate an external ``energy``
        consumes (``output = model(); loss = model.energy(output)``).
      * inference: ``forward(coords)`` returns the matched-pointwise field
        (``decomposition.evaluate(coords)``).

    Args:
        field_layout (FieldLayout): Fresh layout; ``__init__`` registers the
            decomposition's factor fields into it (a layout already holding those
            names raises ``ValueError`` on the duplicate registration).
        decomposition (TensorDecomposition): The separated representation.
        integration_domain (IntegrationDomain): Interpolates all active fields of
            the problem; typically ``IntegrationDomain([*decomposition.assemblies(),
            *other_assemblies])``.
        energy (Callable): Injected callable ``energy(output) -> torch.Tensor``
            (the counterpart of ``FEMModel.loss``), reading fields from the layout.
    """

    def __init__(self, field_layout, decomposition: TensorDecomposition,
                 integration_domain, energy):
        super().__init__()
        self.field_layout = field_layout
        self.decomposition = decomposition
        self.integration_domain = integration_domain
        self.energy = energy
        decomposition.register_into(field_layout)

    def forward(self, coords=None):
        if self.training:
            self.integration_domain.interpolate_all(self.field_layout)
            return self.field_layout
        if coords is None:
            raise ValueError(
                "eval forward requires coords: a (P, n_axes) tensor, one point per row."
            )
        return self.decomposition.evaluate(coords)

    def assemble(self, coords):
        return self.decomposition.assemble(coords)
