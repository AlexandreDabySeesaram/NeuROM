import torch

from neurom.physics.term import Term
from neurom.differential import jacobian_field
from neurom.inner import inner
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase


class ElasticEnergy(Term):
    """Elastic energy term for a displacement field.

    Args:
        field (FieldBase): The field providing the displacement values. Its ``name`` attribute is stored for later lookup.
        modulus (FieldBase | Callable | None): Optional stiffness weighting the
            energy density. Either a :class:`~neurom.fields.field_base.FieldBase`
            interpolated on the *same* quadrature as ``field`` (looked up by name
            in the layout, like the load of
            :class:`~neurom.physics.load_potential.LoadPotential`), or a callable
            ``modulus(x)`` evaluated directly at the quadrature points -- use the
            callable when the modulus is known analytically and you do not want
            to incur its interpolation error. ``None`` (the default) means a unit
            modulus, i.e. the previous behaviour.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the interpolation result from a :class:`~neurom.field_layout.FieldLayout`.
        modulus_name (str | None): Name of the modulus field, when one was given as a field.
        modulus_fn (Callable | None): The modulus callable, when one was given as a callable.

    The elastic energy density for a displacement :math:`u` is given by :math:`\\frac{1}{2}\\,E\\,\\lvert \\nabla u\\rvert^{2}`. This term retrieves the interpolated field from a :class:`~neurom.field_layout.FieldLayout` and computes

    :math:`\\frac{1}{2}\\,E(x)\\,\\big(\\nabla u : \\nabla u\\big)\\,dx`

    where :math:`dx` is the quadrature measure and :math:`E` the modulus (1 if none was given).
    """

    def __init__(self, field: FieldBase, modulus=None) -> None:
        self.field_name = field.name
        self.modulus_name = getattr(modulus, "name", None)
        self.modulus_fn = modulus if self.modulus_name is None else None

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the elastic energy integrand.

        The method performs:
        1. Retrieve the interpolation result for the stored field.
        2. Compute the gradient :math:`\\nabla u`.
        3. Form the inner product :math:`\\nabla u : \\nabla u`, multiply by :math:`0.5`, by the modulus if one was given, and by the quadrature measure :math:`dx`.

        Args:
            field_layout (FieldLayout): Layout providing access to interpolated field data.

        Returns:
            torch.Tensor: Tensor representing the elastic energy density multiplied by the measure at each quadrature point.
        """
        quad_interp_res = field_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure

        # Compute du_dx**2
        du_dx = jacobian_field(x, u)
        inner_product = inner(du_dx, du_dx)
        result = (0.5 * inner_product) * dx

        if self.modulus_name is not None:
            result = field_layout[self.modulus_name].u * result
        elif self.modulus_fn is not None:
            result = self.modulus_fn(x) * result
        return result
