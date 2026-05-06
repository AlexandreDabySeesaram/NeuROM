import torch

from neurom.inner import inner
from neurom.physics.term import Term
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase


class LoadPotential(Term):
    """Potential energy term associated with an external load.

    Args:
        field (FieldBase): The field representing the displacement (or other
            primary variable) to which the load is applied. Its ``name`` attribute
            is stored for later lookup.
        f (FieldBase): The field representing the load density at quadrature points ``x``
            load density at the quadrature points ``x``. Its ``name`` attribute
            is stored for later lookup.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the
            interpolation result from a :class:`~neurom.field_layout.FieldLayout`.
        f_name (str): Name of the associated load density field used to retrieve the
            interpolation result from a :class:`~neurom.field_layout.FieldLayout`.

    The potential energy contributed by an external load ``f`` acting on a
    field ``u`` is
    :math:`-\\int f(x)\\,u(x)\\,dx`.
    This class implements the integrand
    :math:`-\\,f(x)\\,u(x)\\,dx` evaluated at each quadrature point.
    """

    def __init__(self, field: FieldBase, f: FieldBase) -> None:
        """Store the field name and load function.

        Args:
            field (FieldBase): Displacement (or primary) field.
            f (FieldBase): Body force density.
        """
        self.field_name = field.name
        self.f_name = f.name

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the load potential integrand.

        The method performs:
        1. Retrieve the interpolation results for the stored fields.
        2. Evaluate the load function ``f`` at the quadrature points ``x``.
        3. Multiply by the field values ``u`` and the quadrature measure ``dx``
           with a leading minus sign as dictated by the potential energy
           definition.

        Args:
            field_layout (FieldLayout): Layout providing access to interpolated field data.

        Returns:
            torch.Tensor: Tensor representing :math:`-\\,f(x)\\,u(x)\\,dx` at each quadrature point.

        Note:
            No checks are performed to know if the same quadrature rule and the same elements were used to interpolate both ``field`` and ``f``.
        """
        u_interp = field_layout[self.field_name]
        u = u_interp.u
        dx = u_interp.measure

        f_interp = field_layout[self.f_name]
        load = f_interp.u
        inner_product = inner(load, u)
        result = -inner_product * dx
        return result
