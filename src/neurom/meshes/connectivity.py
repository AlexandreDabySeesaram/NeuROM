"""Element-to-node connectivity container."""

import torch.nn as nn


class Connectivity(nn.Module):
    """Mesh topology described by node indices and simplex connectivity.

    The connectivity is described by the node indices and the element
    connectivity, which maps each simplex to its constituent node indices.

    Args:
        nodes_indices (torch.Tensor): The mesh node indices, tensor of shape
            ``(N_nodes,)``.
        element_connectivity (torch.Tensor): The element connectivity, i.e.
            node indices for each simplex. Tensor of shape
            ``(N_elements, N_simplex)`` where ``N_simplex`` is the number of
            points defining the simplex.

    Attributes:
        nodes_indices (torch.Tensor): The mesh node indices, registered as a
            buffer.
        element_connectivity (torch.Tensor): The element connectivity (node
            indices per simplex), registered as a buffer.

    Note:
        No checks are done on whether indices in ``element_connectivity``
        correspond to actual indices in ``nodes_indices``.
    """

    def __init__(self, nodes_indices, element_connectivity):
        super().__init__()

        self.register_buffer("nodes_indices", nodes_indices)
        self.register_buffer("element_connectivity", element_connectivity)

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the connectivity.

        Returns:
            int: The number of nodes.
        """
        return self.nodes_indices.shape[0]

    @property
    def n_elements(self):
        """The number of elements in the connectivity.

        Returns:
            int: The number of elements.
        """
        return self.element_connectivity.shape[0]
