"""Non-local membrane stretching energy of a von Kármán beam."""

import torch

from neurom.physics.term import Term
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase
from neurom.apply import apply
from neurom.math.inner import inner_point
from neurom.math.integrate import integrate
from neurom.math.jacobian import jacobian


class MembraneStretchEnergy(Term):
    r"""Membrane stretching energy :math:`c\,\left(\int w'^2\right)^2`.

    The energy is non-local: it is the square of an integral.  It is written
    as the integrand :math:`c\,S\,w'^2\,dx` with :math:`S = \int w'^2`, so
    that integrating it gives :math:`c\,S^2`.  :math:`S` is kept in the
    autograd graph (not detached), otherwise the gradient would be halved.
    For the von Kármán beam, :math:`c = 1/8`.

    Args:
        field (FieldBase): The field providing the deflection :math:`w`. Its
            ``name`` attribute is stored for later lookup.
        coefficient (float): The coefficient :math:`c`.
    """

    def __init__(self, field: FieldBase, coefficient: float) -> None:
        self.field_name = field.name
        self.coefficient = coefficient

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        r"""Membrane stretching integrand :math:`c\,S\,w'^2\,dx`, shape ``(N_e, N_q, 1)``.

        Args:
            field_layout (FieldLayout): Layout providing the interpolated field.

        Returns:
            torch.Tensor: The smeared integrand; ``integrate`` of it is :math:`c\,S^2`.
        """
        interp = field_layout[self.field_name]
        x = interp.x
        u = interp.u
        dx = interp.measure

        w_prime = jacobian(x, u)  # w'
        stretch = apply(lambda g, m: inner_point(g, g) * m, w_prime, dx).values  # w'^2 dx
        S = integrate(stretch)  # ∫ w'^2 dx  (scalar, kept in the graph)

        return self.coefficient * S * stretch
