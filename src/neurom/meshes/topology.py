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
        n_nodes (int): The number of nodes.
        n_elements (int): The number of elements.

    Note:
        No checks are done on whether indices in `connectivity` correspond to actual indices in `nodes`.
    """

    def __init__(self, nodes, connectivity):
        super().__init__()

        self.register_buffer("nodes", nodes)
        self.register_buffer("connectivity", connectivity)

        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.connectivity.shape[0]
