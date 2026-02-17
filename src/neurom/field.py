import torch
import torch.nn as nn


class Field(nn.Module):
    def __init__(self, mesh, dirichlet_nodes):
        super().__init__()

        n_nodes = mesh.n_nodes

        values = 0.5 * torch.ones(n_nodes, 1)
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        dofs_free[dirichlet_nodes] = False

        self.register_buffer("dofs_free", dofs_free)
        self.values_free = nn.Parameter(values[dofs_free])
        self.register_buffer("values_imposed", torch.zeros((~dofs_free).sum(), 1))

    def full_values(self):
        full = torch.zeros(
            self.dofs_free.shape[0],
            1,
            device=self.values_free.device,
            dtype=self.values_free.dtype,
        )
        full[self.dofs_free] = self.values_free
        full[~self.dofs_free] = self.values_imposed
        return full
