import torch
import torch.nn as nn


from neurom.constraints.constraint import Constraint


class Dirichlet(Constraint):
    """Dirichlet constraint

    Dirichlet constraint owns a tensor representing the nodes which have Dirichlet boundary conditions and their values.

    Args:
        nodes (torch.Tensor): The nodes which have a Dirichelet BC.
        values_imposed (torch.Tensor): The values imposed on those nodes.

    Attributes:
        nodes (torch.Tensor): The nodes which have a Dirichelet BC.
        values_imposed (torch.Tensor): The values imposed on those nodes.
    """

    def __init__(self, nodes, values_imposed):
        super().__init__()
        self.register_buffer("nodes", torch.as_tensor(nodes))
        self.register_buffer("values_imposed", torch.as_tensor(values_imposed))

    def get_dofs_free(self, n_nodes):
        """
        Return boolean mask of free DOFs (True = free, False = constrained)

        Return a tensor of shape n_nodes with indices self.nodes set to False, otherwise set to True.
        """
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        dofs_free[self.nodes] = False
        return dofs_free

    def expand(self, reduced_values, dofs_free):
        """
        Expand the reduced values with the imposed ones

        If `self.values_imposed` is None, the imposed values is set to 0, otherwise it is set to self.values_imposed.

        Returns:
            The fully assembled values over reduced and imposed dofs.
        """
        full = torch.zeros(
            reduced_values.shape[0] + self.values_imposed.shape[0],
            reduced_values.shape[1],
            device=reduced_values.device,
            dtype=reduced_values.dtype,
        )

        full[dofs_free, :] = reduced_values
        full[~dofs_free, :] = self.values_imposed
        return full
