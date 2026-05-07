import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.meshes.mesh import Mesh, Connectivity
from neurom.fields.field import Field
from neurom.fields.trainable_field import TrainableField
from neurom.constraints.no_constraint import NoConstraint

torch.set_default_dtype(torch.float32)


class TestMesh:
    """
    Test Mesh class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_mesh_construction_with_field(self):
        """Test construction of mesh where nodes_positions are defined by a Field"""
        # Number of vertices
        N = 6
        nodes = torch.tensor([0, 1, 2, 3, 4, 5])
        elements = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        connectivity = Connectivity(nodes, elements)
        nodes_positions = torch.tensor([15.0, -1.0, 3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
        x = Field(name="x", connectivity=connectivity, values=nodes_positions)
        mesh = Mesh(connectivity, x)

        # - 1 - Check connectivity
        # Nodes
        assert "nodes_indices" in mesh.connectivity._buffers
        assert isinstance(getattr(mesh.connectivity, "nodes_indices"), torch.Tensor)
        assert mesh.connectivity.nodes_indices.shape == (N,)
        assert (mesh.connectivity.nodes_indices == nodes).all()

        # Connectivity
        assert "element_connectivity" in mesh.connectivity._buffers
        assert isinstance(
            getattr(mesh.connectivity, "element_connectivity"), torch.Tensor
        )
        assert mesh.connectivity.element_connectivity.shape == (N - 1, 2)
        assert (mesh.connectivity.element_connectivity == elements).all()

        # Number of nodes
        assert mesh.connectivity.n_nodes == N

        # Number of elements
        assert mesh.connectivity.n_elements == N - 1

        # - 2- Check nodes_positions
        assert mesh.nodes_positions.name == "x"
        assert "values" in mesh.nodes_positions._buffers
        assert isinstance(getattr(mesh.nodes_positions, "values"), torch.Tensor)
        assert mesh.nodes_positions.full_values() == pytest.approx(
            nodes_positions, rel=self.relative_tolerance
        )

    def test_mesh_construction_with_trainable_field(self):
        """Test construction of mesh where nodes_positions are defined by a TrainableField"""

        # Number of vertices
        N = 6
        nodes = torch.tensor([0, 1, 2, 3, 4, 5])
        elements = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        connectivity = Connectivity(nodes, elements)
        nodes_positions = torch.tensor([15.0, -1.0, 3.0, 7.0, 6.0, -5.0]).unsqueeze(-1)
        x = TrainableField(
            name="x",
            connectivity=connectivity,
            init_values=nodes_positions,
            constraint=NoConstraint(),
        )
        mesh = Mesh(connectivity, x)

        # - 1 - Check connectivity
        # Nodes
        assert "nodes_indices" in mesh.connectivity._buffers
        assert isinstance(getattr(mesh.connectivity, "nodes_indices"), torch.Tensor)
        assert mesh.connectivity.nodes_indices.shape == (N,)
        assert (mesh.connectivity.nodes_indices == nodes).all()

        # Connectivity
        assert "element_connectivity" in mesh.connectivity._buffers
        assert isinstance(
            getattr(mesh.connectivity, "element_connectivity"), torch.Tensor
        )
        assert mesh.connectivity.element_connectivity.shape == (N - 1, 2)
        assert (mesh.connectivity.element_connectivity == elements).all()

        # Number of nodes
        assert mesh.connectivity.n_nodes == N

        # Number of elements
        assert mesh.connectivity.n_elements == N - 1

        # - 2- Check nodes_positions
        assert mesh.nodes_positions.name == "x"
        assert isinstance(mesh.nodes_positions.values_reduced, nn.Parameter)
        assert mesh.nodes_positions.full_values().detach() == pytest.approx(
            nodes_positions, rel=self.relative_tolerance
        )

    def test_incompatible_connectivities(self):
        """Tries to create a mesh with nodes positions having a different connectivity than the one owned by mesh"""
        # Number of vertices
        nodes = torch.tensor([0, 1, 2, 3, 4, 5])
        elements = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        connectivity = Connectivity(nodes, elements)

        # Less nodes positions than nodes
        x = Field(
            name="x",
            connectivity=connectivity,
            values=torch.tensor([15.0, -1.0, 3.0, 7.0, 6.0, -5.0]).unsqueeze(-1),
        )
        other_connectivity = Connectivity(nodes, elements)
        with pytest.raises(ValueError):
            mesh = Mesh(other_connectivity, x)
