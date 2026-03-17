import torch
import torch.nn as nn

from neurom.integrate import integrate


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
        interpolator,
        loss,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field_layout = field_layout
        self.interpolator = interpolator
        self.loss = loss

    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        # Returns a list of QuadratureInterpolationResult
        self.interpolator.interpolate(self.field_layout)
        return self.loss()
