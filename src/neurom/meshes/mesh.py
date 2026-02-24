import torch
import torch.nn as nn

from neurom.meshes.topology import Topology


class Mesh(nn.Module):
    """
    A mesh is defined by:
    * Its topoloy (nodes indices and nodes indices defining connectivity)
    * Its nodes' positions

    Args:
        topology (Topology): The mesh topology.
        nodes_positions (Field | TrainableField): A Field or TrainableField representing nodes positions.

    Attributes:
        topology (Topology): The mesh topology.
        nodes_positions (Field | TrainableField): A Field or TrainableField representing nodes positions.

    Raises:
        ValueError: If there is a different amount of nodes than there are positions, i.e. if  self.nodes_positions.full_values().shape[0] does not match self.nodes_positions.shape[0]
    """

    def __init__(self, topology, nodes_positions):
        super().__init__()

        self.topology = topology
        self.nodes_positions = nodes_positions

        shape_pos = self.nodes_positions.full_values().shape[0]
        shape_nodes = self.topology.nodes.shape[0]
        if shape_pos != shape_nodes:
            raise ValueError(
                f"self.nodes_positions has a different tensor' shape ({shape_pos}) than self.topology.nodes ({shape_nodes})"
            )

    def elements_at(self, x):
        """
        Extract mesh elements ids at which x belongs

        Args:
            x (torch.Tensor): The positions for which we will look for an element of the mesh.

        Returns:
            A tensor with all element ids which own `x`.

        Note:
            Only works for 1D mesh for now.
        """

        # List elements to which `x` belongs to.
        ids = []
        for x_i in x:
            for e, conn in enumerate(self.topology.connectivity):
                x_first = self.nodes_positions.full_values()[conn[0]]
                x_second = self.nodes_positions.full_values()[conn[1]]
                if x_i >= x_first and x_i <= x_second:
                    ids.append(e)
                    break

        element_ids = torch.tensor(ids)
        return element_ids
