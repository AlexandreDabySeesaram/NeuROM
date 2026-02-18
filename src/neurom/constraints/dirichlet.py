import torch
import torch.nn as nn


from neurom.constraints.constraint import Constraint


class Dirichlet(Constraint):
    def __init__(self, dirichlet_nodes, values_imposed=None):
        super().__init__()
        dirichlet_nodes = torch.as_tensor(dirichlet_nodes)
        self.register_buffer("dirichlet_nodes", dirichlet_nodes)
        self.register_buffer("values_imposed", values_imposed)

    def get_dofs_free(self, n_nodes):
        """
        Return boolean mask of free DOFs (True = free, False = constrained)

        Return a tensor of shape n_nodes with indices self.dirichlet_nodes set to False, otherwise set to True.
        """
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        dofs_free[self.dirichlet_nodes] = False
        return dofs_free

    def expand(self, reduced_values, dofs_free):
        """
        Expand the reduced values with the imposed ones

        If `self.values_imposed` is None, the imposed values is set to 0, otherwise it is set to self.values_imposed.

        Returns:
            The fully assembled values over reduced and imposed dofs.
        """
        full = torch.zeros(
            dofs_free.shape[0],
            1,
            device=reduced_values.device,
            dtype=reduced_values.dtype,
        )
        full[dofs_free] = reduced_values

        if self.values_imposed is None:
            imposed = torch.zeros((~dofs_free).sum(), 1, device=full.device)
        else:
            imposed = self.values_imposed.reshape(-1, 1).to(full.device)

        full[~dofs_free] = imposed
        return full
