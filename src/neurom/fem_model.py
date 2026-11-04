"""Top-level FEM model wiring fields, physics and constraints into a trainable module."""

import torch.nn as nn


class FEMModel(nn.Module):
    """Thin orchestration module for a finite-element model.

    Owns all FEM sub-modules so that ``.to(device/dtype)`` and checkpoint
    operations propagate globally.  The ``forward`` pass interpolates all
    fields onto the integration domain and then evaluates the physics loss.

    Attributes:
        mesh: The mesh object describing the geometry and connectivity.
        field_layout (FieldLayout): Container of all registered fields.
        integration_domain: Domain that drives quadrature and interpolation.
        loss: Callable module that computes the scalar loss / energy.
    """

    def __init__(
        self,
        mesh,
        field_layout,
        integration_domain,
        loss,
    ):
        """Initialise the FEM model with its core pipeline components.

        Args:
            mesh: The mesh object describing the geometry and connectivity.
            field_layout (FieldLayout): Container holding all registered fields.
            integration_domain: Domain responsible for quadrature point
                generation and field interpolation.
            loss: A callable ``nn.Module`` that computes the scalar loss or
                potential energy from the interpolated fields.
        """
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field_layout = field_layout
        self.integration_domain = integration_domain
        self.loss = loss

    def forward(self):
        """Run a full forward pass: interpolate all fields then evaluate the loss.

        Calls ``integration_domain.interpolate_all(field_layout)`` to populate
        the field layout with the latest quadrature-point interpolations, then
        delegates to ``self.loss()`` to compute the scalar result.

        Returns:
            torch.Tensor: Scalar loss or potential energy for the current field
            configuration.
        """
        self.integration_domain.interpolate_all(self.field_layout)

        return self.loss()
