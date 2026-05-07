from dataclasses import dataclass
import meshio
import torch
from typing import Tuple, Dict
from neurom.meshes import Connectivity, Mesh


def read_mesh(fname) -> Tuple[Connectivity, Dict[str, torch.tensor]]:
    """Read a mesh and convert it to pytorch arrays

    Read the connectivity of a mesh and parse it as a Connectivity.
    Reads all point data and parse it as a dictionnary of name to pytorch tensor.
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
