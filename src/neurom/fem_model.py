import torch
import torch.nn as nn


class FEMModel(nn.Module):
    """
    Thin orchestration module.

    Responsibilities:
    - Own all FEM submodules so .to(device/dtype) works globally
    - Provide forward() for training / inference
    - Act as checkpoint root
    """

    def __init__(
        self,
        mesh,
        field_layout,
        integration_domain,
        loss,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field_layout = field_layout
        self.integration_domain = integration_domain
        self.loss = loss

    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        self.integration_domain.interpolate_all(self.field_layout)

        return self.loss()
