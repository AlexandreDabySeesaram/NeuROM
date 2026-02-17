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
        evaluator,
        physics,
        integrator,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field = field
        self.evaluator = evaluator
        self.physics = physics
        self.integrator = integrator

    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        x_q, u_q, measure = self.evaluator.evaluate()
        integrand = self.physics.integrand(x_q, u_q)
        loss = self.integrator.integrate(integrand, measure)

        return loss
