from dataclasses import dataclass
import meshio
import torch
from neurom.meshes import Mesh


def area(a, b, c):
    v0 = b - a
    v1 = c - a
    return 0.5 * (v0[0] * v1[1] - v0[1] * v1[0])


def is_valid_mesh(mesh: Mesh):
    """Read a mesh and convert it to pytorch arrays

    Check mesh is valid, i.e. all triangles are properly oriented.
    """
    x = mesh.nodes_positions.at_elements()
    for x_e in x:
        if area(*x_e) <= 0.0:
            return False

    return True
