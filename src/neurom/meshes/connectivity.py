import torch
import torch.nn as nn


class Connectivity(nn.Module):
    """
    The connectivity is described by:
    * The nodes (indices)
    * The connectivity (nodes indices forming simplex)

    Args:
        nodes_indices (torch.Tensor): The mesh nodes indices, tensor of shape (N_nodes).
        element_connectivity (torch.Tensor): The connectivity, i.e. nodes indices. Tensor of shape (N_elements, N_simplex) where N_simplex is the number of points defining the simplex.

    Attributes:
        nodes_indices (torch.Tensor): The mesh nodes indices. Registered in buffer.
        element_connectivity (torch.Tensor): The connectivity, i.e. nodes indices. Registered in buffer.

    Note:
        No checks are done on whether indices in `element_connectivity` correspond to actual indices in `nodes_indices`.
    """

    def __init__(self, nodes_indices, element_connectivity):
        super().__init__()

        self.register_buffer("nodes_indices", nodes_indices)
        self.register_buffer("element_connectivity", element_connectivity)

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the connectivity

        Returns:
            (int) The number of nodes.
        """
        return self.nodes_indices.shape[0]

    @property
    def n_elements(self):
        """The number of elements in the connectivity

        Returns:
            (int) The number of elements.
        """
        return self.element_connectivity.shape[0]
