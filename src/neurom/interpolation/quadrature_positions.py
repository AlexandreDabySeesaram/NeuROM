"""Reference and physical quadrature-point positions for elements."""

from dataclasses import dataclass

from neurom.samplings import QuadratureSampling


@dataclass(frozen=True)
class QuadraturePositions:
    """Immutable container for quadrature-point positions in all coordinate spaces.

    Stores the three coordinate representations needed for autograd-safe field
    interpolation: the original reference coordinates, the mapped physical
    coordinates, and the back-mapped reference coordinates obtained by
    applying the inverse map to the physical positions.

    Attributes:
        xi_ref (QuadratureSampling): Original reference-element coordinates of
            the quadrature points, tensor of shape ``(N_e, N_q, dim)``.
        x_phys (QuadratureSampling): Physical-space coordinates obtained by
            mapping ``xi_ref`` forward through the geometric mapping, tensor
            of shape ``(N_e, N_q, dim)``.
        xi_back (QuadratureSampling): Back-mapped reference coordinates
            obtained by applying the inverse map to ``x_phys``; used as the
            input to shape functions so that autograd can trace gradients
            through the physical positions, tensor of shape
            ``(N_e, N_q, dim)``.
    """

    xi_ref: QuadratureSampling
    x_phys: QuadratureSampling
    xi_back: QuadratureSampling
