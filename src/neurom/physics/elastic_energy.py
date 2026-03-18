import torch

from neurom.physics.term import Term
import neurom.differential as differential
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase


class ElasticEnergy(Term):
    """Elastic energy term for a displacement field.

    Args:
        field (FieldBase): The field providing the displacement values. Its ``name`` attribute is stored for later lookup.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the interpolation result from a :class:`~neurom.field_layout.FieldLayout`.

    The elastic energy density for a displacement :math:`u` is given by :math:`\\frac{1}{2}\\lvert \\nabla u\\rvert^{2}`. This term retrieves the interpolated field from a :class:`~neurom.field_layout.FieldLayout` and computes

    :math:`\\frac{1}{2}\\,\\big(\\nabla u : \\nabla u\\big)\\,dx`

    where :math:`dx` is the quadrature measure.
    """

    def __init__(self, field: FieldBase) -> None:
        self.field_name = field.name

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the elastic energy integrand.

        The method performs:
        1. Retrieve the interpolation result for the stored field.
        2. Compute the gradient :math:`\\nabla u`.
        3. Form the inner product :math:`\\nabla u : \\nabla u` and multiply by :math:`0.5` and the quadrature measure :math:`dx`.

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
        du_dx = differential.jacobian_field(x, u)
        inner = torch.einsum("eq...,eq...->eq", du_dx, du_dx).squeeze()
        result = (0.5 * inner) * dx.squeeze()
        return result
