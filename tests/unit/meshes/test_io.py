import meshio
import numpy as np
import pytest
import torch

# Import library modules
from neurom.meshes.connectivity import Connectivity
from neurom.meshes.mesh import Mesh
from neurom.meshes.io import read_mesh, write_mesh
from neurom.field_layout import FieldLayout
from neurom.fields import Field, TrainableField, ElementField
from neurom.constraints import NoConstraint

torch.set_default_dtype(torch.float32)


# Two counter-clockwise triangles covering the unit square.
POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


class TestReadMesh:
    """Test read_mesh."""

    relative_tolerance: float = 1e-6

    def test_read_mesh(self, tmp_path):
        """Read back a mesh written with meshio and check connectivity and data."""
        tag = np.array([10, 11, 12, 13])
        mesh = meshio.Mesh(
            points=POINTS,
            cells=[("triangle", TRIANGLES)],
            point_data={"tag": tag},
        )
        fname = tmp_path / "mesh.xdmf"
        meshio.write(fname, mesh, file_format="xdmf")

        connectivity, data = read_mesh(fname)

        # Connectivity
        assert isinstance(connectivity, Connectivity)
        assert connectivity.n_nodes == 4
        assert connectivity.n_elements == 2
        assert (connectivity.element_connectivity == torch.tensor(TRIANGLES)).all()

        # Node positions
        assert data["x"].shape == (4, 3)
        assert data["x"].detach().numpy() == pytest.approx(
            POINTS, rel=self.relative_tolerance
        )

        # Point data
        assert "tag" in data["point_data"]
        assert (data["point_data"]["tag"] == torch.tensor(tag)).all()

    def test_read_mesh_without_triangles_raises(self, tmp_path):
        """A mesh that has no triangle cells raises a ValueError."""
        mesh = meshio.Mesh(
            points=POINTS,
            cells=[("line", np.array([[0, 1], [1, 2]]))],
        )
        fname = tmp_path / "lines.xdmf"
        meshio.write(fname, mesh, file_format="xdmf")

        with pytest.raises(ValueError):
            read_mesh(fname)


class TestWriteMesh:
    """Test write_mesh."""

    relative_tolerance: float = 1e-6

    def _build_layout(self):
        """Build a connectivity, mesh and field layout for a unit square."""
        nodes = torch.arange(0, 4)
        elements = torch.tensor(TRIANGLES)
        connectivity = Connectivity(nodes, elements)

        positions = torch.tensor(POINTS[:, 0:2], dtype=torch.float32)

        field_layout = FieldLayout()
        x = field_layout.add(
            Field(name="positions", connectivity=connectivity, values=positions)
        )
        field_layout.add(
            TrainableField(
                name="displacement",
                connectivity=connectivity,
                init_values=0.5 * torch.ones(4, 2),
                constraint=NoConstraint(),
            )
        )
        # One scalar value per element.
        field_layout.add(
            ElementField(name="stress", values=torch.tensor([[1.0], [2.0]]))
        )

        mesh = Mesh(connectivity=connectivity, nodes_positions=x)
        return mesh, field_layout

    def test_write_mesh_roundtrip(self, tmp_path):
        """Write a mesh + fields then read it back with meshio."""
        mesh, field_layout = self._build_layout()

        fname = tmp_path / "out.xdmf"
        write_mesh(fname, mesh, field_layout)

        assert fname.exists()

        result = meshio.read(fname)

        # 2D positions are padded with a zero Z column.
        assert result.points.shape == (4, 3)
        assert result.points == pytest.approx(POINTS, rel=self.relative_tolerance)

        # Connectivity
        assert len(result.cells) == 1
        assert result.cells[0].type == "triangle"
        assert result.cells[0].data == pytest.approx(TRIANGLES)

        # Point data (nodal fields)
        assert "positions" in result.point_data
        assert "displacement" in result.point_data
        assert result.point_data["displacement"] == pytest.approx(
            0.5 * np.ones((4, 2)), rel=self.relative_tolerance
        )

        # Cell data (per-element field)
        assert "stress" in result.cell_data
        assert result.cell_data["stress"][0].reshape(-1) == pytest.approx(
            np.array([1.0, 2.0]), rel=self.relative_tolerance
        )

    def test_write_mesh_creates_parent_directory(self, tmp_path):
        """write_mesh creates the parent directory if it does not exist."""
        mesh, field_layout = self._build_layout()

        fname = tmp_path / "nested" / "dir" / "out.xdmf"
        write_mesh(fname, mesh, field_layout)

        assert fname.exists()
