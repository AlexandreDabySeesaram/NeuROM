import torch
import torch.nn as nn

from neurom.topology import Topology


class Mesh(nn.Module):
    """
    A mesh is defined by:
    * Its topoloy (vertices indices and vertex indices defining connectivity)
    * Its vertices' positions
    """

    def __init__(self, topology, nodes_positions):
        super().__init__()

        self.topology = topology
        self.nodes_positions = nodes_positions
