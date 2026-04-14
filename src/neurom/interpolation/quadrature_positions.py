from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class QuadraturePositions:
    """Dataclass containing the positions at quadrature points

    Attributes:
        xi_ref (torch.Tensor): The reference positions on the element, tensor of shape (N_e, N_q, x_dim).
        u (torch.Tensor): The interpolated positions from self.xi_ref, tensor of shape (N_e, N_q, x_dim).
        measure (torch.Tensor): The reference positions on the element obtained from inverse map from self.x_phys, tensor of shape (N_e, N_q, x_dim).
    """

    xi_ref: torch.Tensor
    x_phys: torch.Tensor
    xi_back: torch.Tensor
