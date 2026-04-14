from dataclasses import dataclass
import torch

from neurom.samplings import QuadratureSampling


@dataclass(frozen=True)
class QuadraturePositions:
    """Dataclass containing the positions at quadrature points

    Attributes:
        xi_ref (QuadratureSampling): The reference positions on the element, tensor of shape (N_e, N_q, x_dim).
        u (QuadratureSampling): The interpolated positions from self.xi_ref, tensor of shape (N_e, N_q, x_dim).
        measure (QuadratureSampling): The reference positions on the element obtained from inverse map from self.x_phys, tensor of shape (N_e, N_q, x_dim).
    """

    xi_ref: QuadratureSampling
    x_phys: QuadratureSampling
    xi_back: QuadratureSampling
