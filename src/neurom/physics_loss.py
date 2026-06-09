"""Loss assembling the total physics energy over a field layout."""

import torch.nn as nn

from neurom.math.integrate import integrate
from neurom.physics.term import Term
from neurom.field_layout import FieldLayout


class PhysicsLoss(nn.Module):
    """Loss module that computes a physics-based variational energy.

    Evaluates the integral of the provided ``physics`` ``Term`` over the domain
    described by the interpolated fields in ``field_layout``.

    Attributes:
        physics (Term): The physics ``Term`` that provides the integrand
            expression.
        field_layout (FieldLayout): The field layout used to look up the
            interpolated field values required by ``physics``.
    """

    def __init__(self, physics: Term, field_layout: FieldLayout):
        """Initialise the physics loss module.

        Args:
            physics (Term): The ``Term`` that defines the integrand of the
                variational energy.
            field_layout (FieldLayout): The layout that maps field names to
                their quadrature interpolation results.
        """
        super().__init__()
        self.physics = physics
        self.field_layout = field_layout

    def forward(self):
        """Compute the physics-based loss by integrating the physics term.

        Evaluates ``self.physics.integrand(self.field_layout)`` to obtain the
        per-quadrature-point integrand tensor and then calls
        :func:`~neurom.math.integrate.integrate` to sum over all elements and
        quadrature points.

        Returns:
            torch.Tensor: Scalar loss value obtained by integrating the physics
            integrand over the domain.
        """
        integrand = self.physics.integrand(self.field_layout)
        result = integrate(integrand)
        return result
