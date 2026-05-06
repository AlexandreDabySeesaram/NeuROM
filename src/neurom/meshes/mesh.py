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
        ValueError: If self.topology differs from nodes_positions.topology.
    """

    def __init__(self, topology, nodes_positions):
        super().__init__()

        self.topology = topology
        self.nodes_positions = nodes_positions

        if self.topology is not self.nodes_positions.topology:
            raise ValueError(
                f"Mesh self.topology does not correspond to self.nodes_positions.topology"
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
