"""Mesh validity checks."""

from neurom.meshes.mesh import Mesh


def signed_area(a, b, c):
    """Compute the signed area of the triangle ``(a, b, c)``.

    The area is positive when the vertices are given in counter-clockwise order
    and negative otherwise.

    Args:
        a: First vertex, tensor of shape ``(2,)``.
        b: Second vertex, tensor of shape ``(2,)``.
        c: Third vertex, tensor of shape ``(2,)``.

    Returns:
        The signed area as a scalar tensor.
    """
    v0 = b - a
    v1 = c - a
    return 0.5 * (v0[0] * v1[1] - v0[1] * v1[0])


def is_valid_mesh(mesh: Mesh) -> bool:
    """Check that all triangles of a mesh are properly oriented.

    A mesh is considered valid when every triangle has a strictly positive
    signed area, i.e. all triangles are oriented counter-clockwise.

    Args:
        mesh (Mesh): The mesh to check.

    Returns:
        ``True`` if all triangles have a strictly positive signed area, ``False``
        otherwise.
    """
    x = mesh.nodes_positions.at_elements()
    for x_e in x:
        if signed_area(*x_e) <= 0.0:
            return False

    return True
