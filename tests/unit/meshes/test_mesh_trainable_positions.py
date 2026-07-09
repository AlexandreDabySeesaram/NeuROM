import torch

from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainablePositions1D

torch.set_default_dtype(torch.float32)


def make_topology(n):
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    return Topology(nodes, elements)


def test_factory_builds_trainable_mesh():
    n = 9
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    mesh = Mesh.with_trainable_positions_1d(top, x)
    assert isinstance(mesh.nodes_positions, TrainablePositions1D)
    assert mesh.topology is top
    assert mesh.nodes_positions.topology is top
    assert mesh.has_trainable_positions
    assert torch.allclose(mesh.nodes_positions.full_values(), x, atol=1e-5)


def test_fixed_mesh_reports_no_trainable_positions():
    n = 9
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    mesh = Mesh(topology=top, nodes_positions=Field(name="pos", topology=top, values=x))
    assert not mesh.has_trainable_positions
