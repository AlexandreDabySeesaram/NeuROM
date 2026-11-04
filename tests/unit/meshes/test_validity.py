import pytest
import torch

# Import library modules
from neurom.meshes.connectivity import Connectivity
from neurom.meshes.mesh import Mesh
from neurom.meshes.validity import is_valid_mesh, signed_area
from neurom.fields.field import Field

torch.set_default_dtype(torch.float32)


def _build_mesh(positions: torch.Tensor, elements: torch.Tensor) -> Mesh:
    """Build a 2D mesh from node positions and triangle connectivity."""
    n_nodes = positions.shape[0]
    nodes = torch.arange(0, n_nodes)
    connectivity = Connectivity(nodes, elements)
    x = Field(name="positions", connectivity=connectivity, values=positions)
    return Mesh(connectivity=connectivity, nodes_positions=x)


class TestSignedArea:
    """Test the signed_area helper."""

    relative_tolerance: float = 1e-9

    def test_counter_clockwise_is_positive(self):
        """A counter-clockwise triangle has a positive signed area."""
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        c = torch.tensor([0.0, 1.0])
        assert signed_area(a, b, c) == pytest.approx(0.5, rel=self.relative_tolerance)

    def test_clockwise_is_negative(self):
        """A clockwise triangle has a negative signed area."""
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        c = torch.tensor([1.0, 0.0])
        assert signed_area(a, b, c) == pytest.approx(-0.5, rel=self.relative_tolerance)

    def test_degenerate_is_zero(self):
        """Three collinear points form a degenerate (zero-area) triangle."""
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        c = torch.tensor([2.0, 0.0])
        assert signed_area(a, b, c) == pytest.approx(0.0, abs=self.relative_tolerance)


class TestIsValidMesh:
    """Test the is_valid_mesh check."""

    def test_valid_mesh(self):
        """All triangles counter-clockwise → the mesh is valid."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Two counter-clockwise triangles covering the unit square.
        elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
        mesh = _build_mesh(positions, elements)
        assert is_valid_mesh(mesh) is True

    def test_invalid_mesh_with_flipped_triangle(self):
        """A single clockwise triangle makes the mesh invalid."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # The second triangle is clockwise (flipped orientation).
        elements = torch.tensor([[0, 1, 2], [0, 3, 2]])
        mesh = _build_mesh(positions, elements)
        assert is_valid_mesh(mesh) is False

    def test_invalid_mesh_with_degenerate_triangle(self):
        """A degenerate (zero-area) triangle is not considered valid."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        elements = torch.tensor([[0, 1, 2]])
        mesh = _build_mesh(positions, elements)
        assert is_valid_mesh(mesh) is False
