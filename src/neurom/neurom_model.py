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
        the filled ``field_layout``** — the intermediate an external ``loss``
        consumes (``output = model(); loss = model.loss(output)``).
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
        loss (Callable): Injected callable ``loss(output) -> torch.Tensor`` (the
            counterpart of ``FEMModel.loss``; the potential energy in a mechanics
            problem), reading fields from the layout.
    """

    def __init__(self, field_layout, decomposition: TensorDecomposition,
                 integration_domain, loss):
        super().__init__()
        self.field_layout = field_layout
        self.decomposition = decomposition
        self.integration_domain = integration_domain
        self.loss = loss
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

    def add_mode_to_optimizer(self, optim, m=None):
        """Add mode ``m``'s parameters to ``optim`` as a new param group.

        Keeps the optimizer wiring in the model: the decomposition only reports
        which tensors make up a mode (:meth:`mode_parameters`), and this method
        does the optimizer-specific ``add_param_group``. The decomposition stays
        agnostic to the optimizer.

        Args:
            optim (torch.optim.Optimizer): Optimizer to enrich.
            m (int, optional): Index of the mode to add. Supports negative
                indexing. Defaults to the last-activated mode.
        """
        params = self.decomposition.mode_parameters(m)
        optim.add_param_group({"params": params})
