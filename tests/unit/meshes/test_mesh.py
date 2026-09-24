import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.meshes.mesh import Mesh
from neurom.meshes.connectivity import Connectivity
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
            Mesh(other_connectivity, x)


def _mesh_1d(n_nodes=5, x_min=0.0, x_max=4.0):
    """A uniform 1-D mesh of ``n_nodes - 1`` elements."""
    nodes = torch.arange(0, n_nodes)
    elements = torch.vstack([torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)]).T
    connectivity = Connectivity(nodes, elements)
    positions = Field(
        name="x",
        connectivity=connectivity,
        values=torch.linspace(x_min, x_max, n_nodes).unsqueeze(-1),
    )
    return Mesh(connectivity, positions)


def _mesh_2d():
    """The unit square split into two triangles: [0, 1, 2] and [0, 2, 3]."""
    nodes = torch.arange(0, 4)
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
    connectivity = Connectivity(nodes, elements)
    positions = Field(
        name="x",
        connectivity=connectivity,
        values=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )
    return Mesh(connectivity, positions)


def _mesh_3d():
    """A single tetrahedron -- enough to have ``dim == 3``."""
    nodes = torch.arange(0, 4)
    elements = torch.tensor([[0, 1, 2, 3]])
    connectivity = Connectivity(nodes, elements)
    positions = Field(
        name="x",
        connectivity=connectivity,
        values=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    return Mesh(connectivity, positions)


class TestElementsAt:
    """Point location: ``Mesh.elements_at`` and its per-dimension backends."""

    def test_interior_points_1d(self):
        """One element index per query point, in a flat (N_pts,) tensor."""
        mesh = _mesh_1d()  # elements [0,1] [1,2] [2,3] [3,4]
        x = torch.tensor([[0.5], [2.5], [3.9]])

        found = mesh.elements_at(x)

        assert found.tolist() == [0, 2, 3]
        # The (N_pts,) shape is part of the contract: it indexes the
        # connectivity row-wise in at_position.
        assert found.shape == (3,)

    def test_point_on_a_shared_node_goes_to_the_first_element_1d(self):
        """A node belongs to both its elements; the lower index wins."""
        mesh = _mesh_1d()

        assert mesh.elements_at(torch.tensor([[2.0]])).tolist() == [1]

    def test_point_outside_the_domain_raises_and_names_it_1d(self):
        mesh = _mesh_1d()

        with pytest.raises(ValueError, match="No element found"):
            mesh.elements_at(torch.tensor([[9.0], [1.0]]))

    def test_interior_points_2d(self):
        mesh = _mesh_2d()
        # (0.8, 0.2) is below the diagonal, (0.2, 0.8) above it.
        x = torch.tensor([[0.8, 0.2], [0.2, 0.8]])

        found = mesh.elements_at(x)

        assert found.tolist() == [0, 1]
        assert found.shape == (2,)

    def test_point_outside_the_domain_raises_2d(self):
        mesh = _mesh_2d()

        with pytest.raises(ValueError, match="No element found"):
            mesh.elements_at(torch.tensor([[5.0, 5.0]]))

    def test_flat_points_are_rejected(self):
        """A (N_pts,) tensor is the shape that broadcasts instead of failing."""
        mesh = _mesh_1d()

        with pytest.raises(ValueError, match="shape"):
            mesh.elements_at(torch.tensor([0.5, 2.5, 3.9]))

    def test_flat_points_are_rejected_even_when_n_pts_equals_n_elements(self):
        """The case the guard exists for: no broadcast error to rely on.

        With four points and four elements the flat tensor pairs up against
        the element axis instead of the point axis, and the old code returned
        ``[0]`` -- one index instead of four, silently wrong.
        """
        mesh = _mesh_1d()  # 5 nodes -> 4 elements
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="shape"):
            mesh.elements_at(x)

        # ... and the same points, correctly shaped, resolve one per element.
        assert mesh.elements_at(x.unsqueeze(-1)).tolist() == [0, 1, 2, 3]

    def test_wrong_coordinate_count_is_rejected(self):
        """2-D points queried against a 1-D mesh."""
        mesh = _mesh_1d()

        with pytest.raises(ValueError, match="dim=1"):
            mesh.elements_at(torch.tensor([[0.5, 0.5]]))

    def test_unsupported_dimension_raises(self):
        """Past 2-D the dispatch used to fall through and return None."""
        mesh = _mesh_3d()

        with pytest.raises(NotImplementedError, match="dim=3"):
            mesh.elements_at(torch.tensor([[0.1, 0.1, 0.1]]))
