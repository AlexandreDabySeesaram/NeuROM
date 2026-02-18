import torch
import torch.nn as nn

from neurom.meshes.topology import Topology


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

    def elements_at(self, x):
        """
        Extract mesh elements ids at which x belongs
        """

        # List elements to which `x` belongs to.
        ids = []
        for x_i in x:
            for e, conn in enumerate(self.topology.conn):
                x_first = self.nodes_positions.full_values()[conn[0]]
                x_second = self.nodes_positions.full_values()[conn[1]]
                if x_i >= x_first and x_i <= x_second:
                    ids.append(e)
                    break

        element_ids = torch.tensor(ids)
        return element_ids
