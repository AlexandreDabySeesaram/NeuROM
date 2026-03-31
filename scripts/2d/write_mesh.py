import meshio
from pathlib import Path
import numpy as np
import torch
from typing import Tuple

from neurom.field_layout import FieldLayout
from neurom.meshes import Mesh


def write_mesh(fname: Path, mesh: Mesh, field_layout: FieldLayout) -> None:
    """Write a mesh together with per‑node field data to an XDMF file.

    Args:
        mesh (Mesh): The mesh.
        field_layout (FieldLayout): The field layout to save.
        filename epathlib.Path): File name to write the mesh.

    The function relies on *meshio* for the actual file writing.
    """
    # Convert tensors to NumPy – meshio works with plain NumPy arrays.
    points_np: np.ndarray = mesh.nodes_positions.full_values().detach().cpu().numpy()

    # meshio expects a (N, dim) array; ensure a 3‑D shape for XDMF
    if points_np.shape[1] == 2:  # 2‑D case → pad with zero Z
        points_np = np.column_stack([points_np, np.zeros(points_np.shape[0])])

    # Build the cell block(s).
    connectivity_np: np.ndarray = mesh.topology.connectivity.detach().cpu().numpy()
    cells = [("triangle", connectivity_np)]

    # Gather point‑wise data from the supplied fields.
    point_data: dict[str, np.ndarray] = {}
    # TODO: add iteration over field_layout
    for field_name, field in field_layout._fields.items():
        values = field.full_values()
        # Ensure shape (N, ?) – flatten scalar fields to (N,)
        values_np = values.detach().cpu().numpy()
        if values_np.ndim == 2 and values_np.shape[1] == 1:
            values_np = values_np.squeeze(1)
        point_data[field_name] = values_np

    # Write the file with meshio.
    mesh = meshio.Mesh(points=points_np, cells=cells, point_data=point_data)
    fname.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(fname, mesh, file_format="xdmf")
    print(f"Exported mesh + fields to {fname}")
