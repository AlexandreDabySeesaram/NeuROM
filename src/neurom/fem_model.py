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
        field,
        interpolator,
        physics,
        integrator,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field = field
        self.interpolator = interpolator
        self.physics = physics
        self.integrator = integrator

    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        result = self.interpolator.interpolate()
        integrand = self.physics.integrand(result.x, result.u)
        loss = self.integrator.integrate(integrand, result.measure)

        return loss
