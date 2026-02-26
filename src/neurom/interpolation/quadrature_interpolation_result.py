from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class QuadratureInterpolationResult:
    """Dataclass containing what's needed to integrate a field on quadrature points

    Attributes:
        x (torch.Tensor): The physical positions, tensor of shape (N_e, N_q, x_dim).
        u (torch.Tensor): The interpolated field, tensor of shape (N_e, N_q, u_dim).
        measure (torch.Tensor): The measure for integration, i.e. dx * w, tensor of shape (N_e, N_q, 1).
    """

    x: torch.Tensor
    u: torch.Tensor
    measure: torch.Tensor
