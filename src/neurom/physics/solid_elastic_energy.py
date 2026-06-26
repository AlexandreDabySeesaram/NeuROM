"""Constitutive-law based solid elastic strain-energy physics term."""

import torch

from neurom.physics.term import Term
from neurom.math.inner import inner_point
from neurom.field_layout import FieldLayout
from neurom.fields.field_base import FieldBase
from neurom.apply import apply


class SolidElasticEnergy(Term):
    """Elastic energy term for a solid mechanics displacement field.

    Computes the strain-energy density :math:`\\frac{1}{2}\\,\\sigma : \\epsilon`
    integrated over the domain, where :math:`\\epsilon` is the strain obtained
    from a caller-supplied ``strain`` function and :math:`\\sigma` is the
    corresponding stress obtained from a caller-supplied ``stress_point``
    function.

    Args:
        field (FieldBase): The field providing the displacement values. Its
            ``name`` attribute is stored for later lookup.
        strain (callable): Function ``(x, u) -> Sampling`` that computes the
            strain tensor from coordinates ``x`` and displacement ``u``.
        stress_point (callable): Function ``(eps) -> torch.Tensor`` that
            computes the Cauchy stress tensor at a single quadrature point
            given the strain tensor ``eps``.

    Attributes:
        field_name (str): Name of the associated field used to retrieve the
            interpolation result from a
            :class:`~neurom.field_layout.FieldLayout`.
        strain (callable): Strain computation function supplied at
            construction time.
        stress_point (callable): Point-wise stress computation function
            supplied at construction time.
    """

    def __init__(self, field: FieldBase, strain, stress_point) -> None:
        """Store the field name and constitutive-law callables.

        Args:
            field (FieldBase): The field providing the displacement values.
                Its ``name`` attribute is stored for later lookup.
            strain (callable): Function ``(x, u) -> Sampling`` that computes
                the strain tensor sampling from coordinates and displacement.
            stress_point (callable): Function ``(eps) -> torch.Tensor`` that
                computes the Cauchy stress tensor at a single quadrature point.
        """
        self.field_name = field.name
        self.strain = strain
        self.stress_point = stress_point

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:
        """Compute the elastic energy integrand.

        Args:
            field_layout (FieldLayout): Layout providing access to interpolated
                field data.

        Returns:
            torch.Tensor: Tensor representing the elastic energy density
            multiplied by the measure at each quadrature point.
        """
        quad_interp_res = field_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure

        epsilon = self.strain(x, u)

        def elastic_energy_point(eps, dx):
            return 0.5 * inner_point(self.stress_point(eps), eps) * dx

        return apply(elastic_energy_point, epsilon, dx).values
