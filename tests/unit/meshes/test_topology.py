import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.meshes.topology import Topology

torch.set_default_dtype(torch.float32)


def test_topology():
    """
    Test Topology class

    Test that passing the nodes indices and the connectivity, we get the expected attributes
    """
    # Number of vertices
    N = 6
    nodes = torch.tensor([0, 1, 2, 3, 4, 5])
    elements = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    topology = Topology(nodes, elements)

    # Nodes
    assert "nodes" in topology._buffers
    assert isinstance(getattr(topology, "nodes"), torch.Tensor)
    assert topology.nodes.shape == (N,)
    assert (topology.nodes == nodes).all()

    # Connectivity
    assert "connectivity" in topology._buffers
    assert isinstance(getattr(topology, "connectivity"), torch.Tensor)
    assert topology.connectivity.shape == (N - 1, 2)
    assert (topology.connectivity == elements).all()

    # Number of nodes
    assert topology.n_nodes == N

    # Number of elements
    assert topology.n_elements == N - 1
