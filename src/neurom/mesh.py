import torch
import torch.nn as nn


class Mesh(nn.Module):
    """
    A mesh is defined by:
    * Its vertices (indices)
    * Its connectivity (vertex indices forming simplex)
    * Its vertices' positions
    """

    def __init__(self, nodes, connectivity):
        super().__init__()

        self.register_buffer("nodes_positions", nodes)
        self.register_buffer("conn", connectivity)

        self.n_nodes = nodes.shape[0]
        self.n_elements = connectivity.shape[0]

        element_ids = torch.arange(self.conn.size(0))
        element_nodes_ids = self.conn[element_ids, :].T
        element_nodes_ids = element_nodes_ids.t()[:, :, None]

        self.register_buffer("element_nodes_ids", element_nodes_ids)

        element_nodes_positions = self.nodes_positions[self.conn]

        self.register_buffer(
            "element_nodes_positions",
            element_nodes_positions.to(self.nodes_positions.dtype),
        )
