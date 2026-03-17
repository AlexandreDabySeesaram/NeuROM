import torch
import torch.nn as nn

from neurom.integrate import integrate
from neurom.physics.term import Term
from neurom.field_layout import FieldLayout


class PhysicsLoss(nn.Module):
    """Provides a loss function based on physics

    This integrates the terms present in the ``physics`` Term with the fields in the provided ``field_layout``.

    Args:
        physics (Term): The physics providing the integrand.
        field_layout (FieldLayout): The field layout in which we will look for the actual fields in the integrand.
    Attributes:
        physics (Term): The physics providing the integrand.
        field_layout (FieldLayout): The field layout in which we will look for the actual fields in the integrand.
    """

    def __init__(self, physics: Term, field_layout: FieldLayout):
        super().__init__()
        self.physics = physics
        self.field_layout = field_layout

    def forward(self):
        integrand = self.physics.integrand(self.field_layout)
        return integrate(integrand)
