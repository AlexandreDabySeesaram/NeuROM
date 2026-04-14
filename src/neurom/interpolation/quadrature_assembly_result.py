from dataclasses import dataclass
import torch

from neurom.samplings import QuadratureSampling


@dataclass(frozen=True)
class QuadratureAssemblyResult:
    """The result of a QuadratureAssembly

    This contains what's needed to integrate a field on quadrature points

    Attributes:
        x (QuadratureSampling): The physical positions, tensor of shape (N_e, N_q, x_dim).
        u (QuadratureSampling): The interpolated field, tensor of shape (N_e, N_q, u_dim).
        measure (QuadratureSampling): The measure for integration, i.e. dx * w, tensor of shape (N_e, N_q, 1).
    """

    x: QuadratureSampling
    u: QuadratureSampling
    measure: QuadratureSampling
