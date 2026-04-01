import torch

from neurom.physics.term import Term
from neurom.physics.tensors import cauchy_stress
from neurom.inner import inner
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase


class SolidElasticEnergy(Term):
    """Elastic energy term for a displacement field.

    Args:
        field (FieldBase): The field providing the displacement values. Its ``name`` attribute is stored for later lookup.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the interpolation result from a :class:`~neurom.field_layout.FieldLayout`.

    The elastic energy density for a displacement :math:`u` is given by :math:`\\frac{1}{2}\\lvert \\nabla u\\rvert^{2}`. This term retrieves the interpolated field from a :class:`~neurom.field_layout.FieldLayout` and computes

    :math:`\\frac{1}{2}\\,\\big(\\nabla u : \\nabla u\\big)\\,dx`

    where :math:`dx` is the quadrature measure.
    """

    def __init__(self, field: FieldBase, strain, constitutive_law) -> None:
        self.field_name = field.name
        self.strain = strain
        self.constitutive_law = constitutive_law

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the elastic energy integrand.

        Args
            field_layout (FieldLayout): Layout providing access to interpolated field data.

        Returns:
            torch.Tensor: Tensor representing the elastic energy density multiplied by the measure at each quadrature point.
        """
        quad_interp_res = field_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure
        epsilon = self.strain(x, u)
        sigma = cauchy_stress(x, u, self.strain, self.constitutive_law)

        # Compute sigma(u):epsilon(u)
        inner_product = inner(sigma, epsilon)
        result = (0.5 * inner_product) * dx
        return result
