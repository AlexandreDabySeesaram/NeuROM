import torch
import torch.nn as nn


class Topology(nn.Module):
    """
    The topology is described by:
    * The vertices (indices)
    * The connectivity (vertex indices forming simplex)
    """

    def __init__(self, nodes, connectivity):
        super().__init__()

        self.nodes = nodes
        self.register_buffer("conn", connectivity)

        self.n_nodes = self.nodes.shape[0]
        self.n_elements = connectivity.shape[0]

        element_ids = torch.arange(self.conn.size(0))
        element_nodes_ids = self.conn[element_ids, :].T
        element_nodes_ids = element_nodes_ids.t()[:, :, None]

        self.register_buffer("element_nodes_ids", element_nodes_ids)
