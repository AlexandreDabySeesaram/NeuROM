"""Non-local membrane stretching energy of a von Kármán beam."""

import torch

from neurom.physics.term import Term
from neurom.field_layout import FieldLayout
from neurom.math.integrate import integrate


class MembraneStretchEnergy(Term):
    """Membrane stretching energy :math:`\\frac{1}{8}\\left(\\int w'^2\\right)^2`.

    The energy is non-local: it is the square of an integral.  It is written
    as the integrand :math:`\\frac{S}{8}\\,w'^2\\,dx` with
    :math:`S = \\int w'^2`, so that integrating it gives :math:`S^2/8`.
    :math:`S` is kept in the autograd graph (not detached), otherwise the
    gradient would be halved.

    Args:
        stretch (Term): Term whose integrand is :math:`w'^2\\,dx`,
            e.g. ``SolidElasticEnergy(w, strain=jacobian, stress_point=lambda eps: 2 * eps)``.
    """

    def __init__(self, stretch: Term):
        self.stretch = stretch

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the membrane stretching integrand :math:`\\frac{S}{8}\\,w'^2\\,dx`.

        Args:
            field_layout (FieldLayout): Layout providing access to interpolated
                field data.

        Returns:
            torch.Tensor: Integrand of shape ``(N_e, N_q)``.
        """
        stretch = self.stretch.integrand(field_layout)  # w'^2 dx
        S = integrate(stretch)
        return S / 8 * stretch
