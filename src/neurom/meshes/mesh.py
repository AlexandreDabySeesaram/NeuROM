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

        # One element per query point, first (lowest-id) match -- vectorised over
        # both points and elements. The Python double loop this replaces was
        # O(points x elements) and stalled at tens of thousands of query points
        # (the 627-point parameter grid times the space samples).
        connectivity = self.topology.connectivity
        positions = self.nodes_positions.full_values().reshape(-1)
        left = positions[connectivity[:, 0]]  # (E,) element left ends
        right = positions[connectivity[:, 1]]  # (E,) element right ends

        xf = x.reshape(-1, 1)  # (P, 1)
        inside = (xf >= left.reshape(1, -1)) & (xf <= right.reshape(1, -1))  # (P, E)
        # argmax returns the first True per row (== lowest element id, the old
        # loop's `break`); shared interior nodes resolve to the lower element.
        return inside.to(torch.uint8).argmax(dim=1)
