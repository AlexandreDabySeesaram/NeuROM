import torch
import torch.nn as nn


class Topology(nn.Module):
    """
    The topology is described by:
    * The nodes (indices)
    * The connectivity (nodes indices forming simplex)

    Args:
        nodes (torch.Tensor): The mesh nodes indices, tensor of shape (N_nodes).
        connectivity (torch.Tensor): The connectivity, i.e. nodes indices. Tensor of shape (N_elements, N_simplex) where N_simplex is the number of points defining the simplex.

    Attributes:
        nodes (torch.Tensor): The mesh nodes indices. Registered in buffer.
        connectivity (torch.Tensor): The connectivity, i.e. nodes indices. Registered in buffer.

    Note:
        No checks are done on whether indices in `connectivity` correspond to actual indices in `nodes`.
    """

    def __init__(self, nodes, connectivity):
        super().__init__()

        self.register_buffer("nodes", nodes)
        self.register_buffer("connectivity", connectivity)

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the topology

        Returns:
            (int) The number of nodes.
        """
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        """The number of elements in the topology

        Returns:
            (int) The number of elements.
        """
        return self.connectivity.shape[0]
