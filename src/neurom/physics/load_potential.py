from neurom.physics.term import Term
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase
import torch


class LoadPotential(Term):
    """Potential energy term associated with an external load.

    Args:
        field (FieldBase): The field representing the displacement (or other
            primary variable) to which the load is applied. Its ``name`` attribute
            is stored for later lookup.
        f (Callable[[torch.Tensor], torch.Tensor]): A callable that evaluates the
            load density at the quadrature points ``x``. It must return a tensor
            compatible with the field values ``u``.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the
            interpolation result from a :class:`~neurom.field_layout.FieldLayout`.
        f (Callable): The load density function.

    The potential energy contributed by an external load ``f`` acting on a
    field ``u`` is
    :math:`-\\int f(x)\\,u\\,dx`.
    This class implements the integrand
    :math:`-\\,f(x)\\,u\\,dx` evaluated at each quadrature point.
    """

    def __init__(self, field: FieldBase, f) -> None:
        """Store the field name and load function.

        Args:
            field (FieldBase): Displacement (or primary) field.
            f (Callable[[torch.Tensor], torch.Tensor]): Load density function.
        """
        self.field_name = field.name
        self.f = f

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the load potential integrand.

        The method performs:
        1. Retrieve the interpolation result for the stored field.
        2. Evaluate the load function ``f`` at the quadrature points ``x``.
        3. Multiply by the field values ``u`` and the quadrature measure ``dx``
           with a leading minus sign as dictated by the potential energy
           definition.

        Args:
            field_layout (FieldLayout): Layout providing access to interpolated field data.

        Returns:
            torch.Tensor: Tensor representing :math:`-\\,f(x)\\,u\\,dx` at each quadrature point.
        """
        quad_interp_res = field_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure

        return -(self.f(x) * u).squeeze() * dx
