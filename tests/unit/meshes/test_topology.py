import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.meshes.connectivity import Connectivity

torch.set_default_dtype(torch.float32)


def test_connectivity():
    """
    Test Connectivity class

    Test that passing the nodes indices and the connectivity, we get the expected attributes
    """
    # Number of vertices
    N = 6
    nodes = torch.tensor([0, 1, 2, 3, 4, 5])
    elements = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    connectivity = Connectivity(nodes, elements)

    # Nodes
    assert "nodes_indices" in connectivity._buffers
    assert isinstance(getattr(connectivity, "nodes_indices"), torch.Tensor)
    assert connectivity.nodes_indices.shape == (N,)
    assert (connectivity.nodes_indices == nodes).all()

    # Connectivity
    assert "element_connectivity" in connectivity._buffers
    assert isinstance(getattr(connectivity, "element_connectivity"), torch.Tensor)
    assert connectivity.element_connectivity.shape == (N - 1, 2)
    assert (connectivity.element_connectivity == elements).all()

    # Number of nodes
    assert connectivity.n_nodes == N

    # Number of elements
    assert connectivity.n_elements == N - 1
