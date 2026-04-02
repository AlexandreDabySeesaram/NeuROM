import torch
import torch.nn as nn

from neurom.meshes.topology import Topology


def is_in_triangle(p, a, b, c):
    p = torch.cat([p, torch.zeros(1, dtype=p.dtype, device=p.device)])
    a = torch.cat([a, torch.zeros(1, dtype=a.dtype, device=a.device)])
    b = torch.cat([b, torch.zeros(1, dtype=b.dtype, device=b.device)])
    c = torch.cat([c, torch.zeros(1, dtype=c.dtype, device=c.device)])

    # a ---- b
    #  \    /
    #   \  /
    #    c
    n = torch.tensor([0.0, 0.0, 1.0])
    if torch.dot(p - a, torch.linalg.cross(n, b - a)) < 0.0:
        # print("First")
        return False
    if torch.dot(p - b, torch.linalg.cross(n, c - b)) < 0.0:
        # print("Second")
        return False
    if torch.dot(p - c, torch.linalg.cross(n, a - c)) < 0.0:
        # print("Third")
        return False
    return True


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
        self.dim = self.nodes_positions.dim

        if self.topology is not self.nodes_positions.topology:
            raise ValueError(
                f"Mesh self.topology does not correspond to self.nodes_positions.topology"
            )

    @property
    def n_nodes(self):
        return self.topology.n_nodes

    @property
    def n_elements(self):
        return self.topology.n_elements

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
        if self.dim == 1:
            for x_i in x:
                for e, conn in enumerate(self.topology.connectivity):
                    x_first = self.nodes_positions.full_values()[conn[0]]
                    x_second = self.nodes_positions.full_values()[conn[1]]
                    if x_i >= x_first and x_i <= x_second:
                        ids.append(e)
                        break
        if self.dim == 2:
            for x_i in x:
                num = len(ids)
                for e, conn in enumerate(self.topology.connectivity):
                    x_0, x_1, x_2 = self.nodes_positions.full_values()[conn]
                    if is_in_triangle(x_i.squeeze(), x_0, x_1, x_2):
                        ids.append(e)
                        break
                if num == len(ids):
                    raise ValueError(
                        f"Did not found element corrsponding to position '{x_i}'"
                    )

        element_ids = torch.tensor(ids)
        return element_ids
