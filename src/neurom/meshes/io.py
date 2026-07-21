"""Mesh input/output utilities.

Read meshes from disk into a :class:`~neurom.meshes.connectivity.Connectivity`
plus the associated point data, and write a :class:`~neurom.meshes.mesh.Mesh`
together with its field data back to disk. The actual file parsing/writing is
delegated to *meshio*.
"""

from pathlib import Path
from typing import Dict, Tuple

import meshio
import numpy as np
import torch

from neurom.field_layout import FieldLayout
from neurom.fields import ElementField, Field, TrainableField
from neurom.meshes.connectivity import Connectivity
from neurom.meshes.mesh import Mesh


def read_mesh(fname) -> Tuple[Connectivity, Dict[str, torch.Tensor]]:
    """Read a mesh from disk and convert it to pytorch arrays.

    Reads the connectivity of a mesh and parses it as a
    :class:`~neurom.meshes.connectivity.Connectivity`. Reads all point data and
    parses it as a dictionary mapping each name to a pytorch tensor.

    Args:
        fname: Path to the mesh file (any format understood by *meshio*).

    Returns:
        A tuple ``(connectivity, data)`` where ``connectivity`` is a
        :class:`~neurom.meshes.connectivity.Connectivity` built from the
        ``triangle`` cells and ``data`` is a dictionary holding the node
        positions under ``"x"`` (shape ``(N_nodes, 3)``) and the per-node data
        under ``"point_data"``.

    Raises:
        ValueError: If no ``triangle`` elements are found in the mesh.
    """

    # ---- Load the mesh ----
    mesh = meshio.read(fname)

    # ---- Extract node positions ----
    points = torch.tensor(mesh.points)  # shape (N, 3)
    n_nodes = points.shape[0]
    data = {}
    data["x"] = points

    # ---- Extract connectivity ----
    triangles = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = torch.tensor(cell_block.data)
            break

    if triangles is None:
        raise ValueError("No 'triangle' elements found in the mesh.")

    # Parse point data
    data["point_data"] = {}
    for name, point_data in mesh.point_data.items():
        data["point_data"][name] = torch.tensor(point_data)

    # Nodes ids
    nodes = torch.arange(0, n_nodes)

    # Initialize connectivity
    connectivity = Connectivity(nodes, triangles)

    # Return connectivity
    return (connectivity, data)


def write_mesh(fname: Path, mesh: Mesh, field_layout: FieldLayout) -> None:
    """Write a mesh together with per-node and per-element field data.

    The fields held by ``field_layout`` are dispatched to point data
    (:class:`~neurom.fields.field.Field` and
    :class:`~neurom.fields.trainable_field.TrainableField`) or cell data
    (:class:`~neurom.fields.element_field.ElementField`) and written to an XDMF
    file via *meshio*.

    Args:
        fname (Path): File name to write the mesh to.
        mesh (Mesh): The mesh whose topology and node positions are written.
        field_layout (FieldLayout): The field layout whose fields are written.

    Raises:
        TypeError: If a field in ``field_layout`` is of an unsupported type.
    """

    # Convert tensors to NumPy – meshio works with plain NumPy arrays.
    points_np: np.ndarray = mesh.nodes_positions.full_values().detach().cpu().numpy()

    # meshio expects a (N, dim) array; ensure a 3‑D shape for XDMF
    if points_np.shape[1] == 2:  # 2‑D case → pad with zero Z
        points_np = np.column_stack([points_np, np.zeros(points_np.shape[0])])

    # Build the cell block(s).
    connectivity_np: np.ndarray = (
        mesh.connectivity.element_connectivity.detach().cpu().numpy()
    )
    cells = [("triangle", connectivity_np)]

    # Containers for mesh data.
    point_data = {}
    cell_data = {}

    # Fill mesh data
    for name, field in field_layout._fields.items():
        match field:
            case Field():
                point_data[name] = field.full_values().detach().cpu().numpy()
            case TrainableField():
                point_data[name] = field.full_values().detach().cpu().numpy()
            case ElementField():
                arr = field.values.detach().cpu().numpy()
                cell_data[name] = [arr]

            case _:
                raise TypeError(f"{field}")

    mesh_out = meshio.Mesh(
        points=points_np, cells=cells, point_data=point_data, cell_data=cell_data
    )
    fname.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(fname, mesh_out, file_format="xdmf")

    print(f"Exported mesh + fields to {fname}")
