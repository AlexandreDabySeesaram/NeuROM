"""Result container for quadrature-point assembly (positions, values, measure)."""

from dataclasses import dataclass

from neurom.samplings import QuadratureSampling


@dataclass(frozen=True)
class QuadratureAssemblyResult:
    """The result of a ``QuadratureAssembly``.

    Bundles together everything needed to numerically integrate a field over
    quadrature points: the physical positions, the interpolated field values,
    and the integration measure.

    Attributes:
        x (QuadratureSampling): Physical positions at the quadrature points,
            tensor of shape ``(N_e, N_q, x_dim)``.
        u (QuadratureSampling): Interpolated field values at the quadrature
            points, tensor of shape ``(N_e, N_q, u_dim)``.
        measure (QuadratureSampling): Integration measure at each quadrature
            point (quadrature weight times the absolute Jacobian determinant),
            tensor of shape ``(N_e, N_q, 1)``.
    """

    x: QuadratureSampling
    u: QuadratureSampling
    measure: QuadratureSampling
