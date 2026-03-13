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
        field,
        interpolator,
        physics,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field = field
        self.interpolator = interpolator
        self.physics = physics

    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        result = self.interpolator.interpolate()
        layout = {self.field.name: result}

        def physics_loss(phys, lay):
            integrand = phys.integrand(layout)
            return integrate(integrand)

        return physics_loss(self.physics, layout)
