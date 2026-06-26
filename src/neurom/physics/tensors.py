"""Tensor operations for solid mechanics (stress, strain, invariants)."""

import torch

from neurom.math.jacobian import jacobian
from neurom.math.inner import inner_point
from neurom.math.transpose import transpose_point
from neurom.math.identity import identity_point
from neurom.math.trace import trace_point

from neurom.samplings import Sampling
from neurom.apply import apply


def linear_elastic_stress_point(
    strain: torch.Tensor,
    lame_lambda: float,
    lame_mu: float,
) -> torch.Tensor:
    """Compute the linear elastic stress tensor at a single quadrature point.

    Evaluates the constitutive relation
    :math:`\\sigma = \\lambda\\,\\text{tr}(\\epsilon)\\,I + 2\\mu\\,\\epsilon`.

    Args:
        strain (torch.Tensor): Strain tensor of shape ``(d, d)``.
        lame_lambda (float): First Lame parameter :math:`\\lambda`.
        lame_mu (float): Second Lame parameter (shear modulus) :math:`\\mu`.

    Returns:
        torch.Tensor: Cauchy stress tensor of shape ``(d, d)``.
    """
    return (
        lame_lambda * trace_point(strain) * identity_point(strain)
        + 2.0 * lame_mu * strain
    )


def linear_elastic_stress(
    strain: Sampling, lame_lambda: float, lame_mu: float
) -> Sampling:
    """Apply the linear elastic constitutive law over a full sampling.

    Calls :func:`linear_elastic_stress_point` element-wise on every quadrature
    point in ``strain`` via :func:`~neurom.apply.apply`.

    Args:
        strain (Sampling): Strain tensor sampling over all quadrature points.
        lame_lambda (float): First Lame parameter :math:`\\lambda`.
        lame_mu (float): Second Lame parameter (shear modulus) :math:`\\mu`.

    Returns:
        Sampling: Cauchy stress tensor sampling with the same structure as
        ``strain``.
    """

    return apply(
        linear_elastic_stress_point,
        strain,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
    )


def green_lagrange_strain(x: Sampling, u: Sampling) -> Sampling:
    """Compute the Green-Lagrange strain tensor from a displacement field.

    Computes the symmetric part of the displacement gradient,
    :math:`\\epsilon = \\frac{1}{2}(\\nabla u + (\\nabla u)^T)`, which is the
    linearised (small-strain) strain measure.  Both ``x`` and ``u`` must be
    instances of the same :class:`~neurom.samplings.Sampling` subclass.

    Args:
        x (Sampling): Quadrature-point coordinates sampling.
        u (Sampling): Displacement field sampling at the same quadrature points.

    Returns:
        Sampling: Symmetric strain tensor sampling with the same type and batch
        shape as the inputs.

    Raises:
        AssertionError: If ``x`` and ``u`` are not instances of the same
            ``Sampling`` subclass.
    """
    assert type(x) is type(u), (
        f"x and u must be of the same Sampling type but got x of type '{type(x)}' and u of type '{type(u)}'"
    )

    grad = jacobian(x, u)
    return apply(lambda du_dx: 0.5 * (du_dx + transpose_point(du_dx)), grad)


def stress_deviator_point(stress: torch.Tensor) -> torch.Tensor:
    """Compute the deviatoric part of a stress tensor at a single point.

    Subtracts the isotropic (hydrostatic) component:
    :math:`s = \\sigma - \\frac{1}{3}\\,\\text{tr}(\\sigma)\\,I`.

    Args:
        stress (torch.Tensor): Cauchy stress tensor of shape ``(d, d)``.

    Returns:
        torch.Tensor: Deviatoric stress tensor of shape ``(d, d)``.
    """

    return stress - 1.0 / 3.0 * trace_point(stress) * identity_point(stress)


def stress_deviator(stress: Sampling) -> torch.Tensor:
    """Compute the deviatoric part of the stress tensor over a full sampling.

    Calls :func:`stress_deviator_point` element-wise on every quadrature point
    in ``stress`` via :func:`~neurom.apply.apply`.

    Args:
        stress (Sampling): Cauchy stress tensor sampling whose field dimensions
            are ``(d, d)``.

    Returns:
        torch.Tensor: Deviatoric stress tensor sampling with the same structure
        as the input.
    """
    return apply(stress_deviator_point, stress)


def stress_von_mises_point(stress_dev):
    """Compute the von Mises equivalent stress at a single quadrature point.

    Evaluates :math:`\\sigma_{vM} = \\sqrt{\\frac{3}{2}\\,s:s}` where
    :math:`s` is the deviatoric stress tensor.

    Args:
        stress_dev (torch.Tensor): Deviatoric stress tensor of shape ``(d, d)``.

    Returns:
        torch.Tensor: Scalar von Mises equivalent stress.
    """
    return torch.sqrt(1.5 * inner_point(stress_dev, stress_dev))


def stress_von_mises(stress_dev):
    """Compute the von Mises equivalent stress over a full sampling.

    Calls :func:`stress_von_mises_point` element-wise on every quadrature point
    in ``stress_dev`` via :func:`~neurom.apply.apply`.

    Args:
        stress_dev (Sampling): Deviatoric stress tensor sampling whose field
            dimensions are ``(d, d)``.

    Returns:
        torch.Tensor: Von Mises equivalent stress sampling with the batch
        shape of the input.
    """
    return apply(stress_von_mises_point, stress_dev)
