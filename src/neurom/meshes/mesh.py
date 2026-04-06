import torch
import torch.nn as nn

from neurom.meshes.topology import Topology


def is_in_triangle(pts, vertices):
    """Find if points are in a triangle defined by its vertices
    Args:
        pts (torch.Tensor): The points we want to check (N_pts, 2).
        vertices (torch.Tensor): The triangle vertices (3, N_e)
    Returns:
        A boolean torch.Tensor of shape (N_pts, N_e)
    """
    # a ---- c
    #  \   /
    #   \ /
    #    b
    # Get individual vertices positions
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Expand for broadcasting: (N_pts, 1, 2) vs (1, N_e, 2)
    pts = pts[:, None, :]  # (N_pts, 1, 2)
    a = a[None, :, :]  # (1, N_e, 2)
    b = b[None, :, :]
    c = c[None, :, :]

    # 2D cross product (p-a) x (b-a): scalar z-component
    def cross2d(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    # (N_pts, N_e)
    d0 = cross2d(pts - a, b - a)
    d1 = cross2d(pts - b, c - b)
    d2 = cross2d(pts - c, a - c)

    return (d0 <= 0) & (d1 <= 0) & (d2 <= 0)  # (N_pts, N_e)


def elements_at_2d(x, nodes_positions, connectivity):
    """Find all elements corresponding in which pts lie

    Args:
        x (torch.Tensor): The points to look for (N_pts, 2)
        nodes_positions (torch.Tensor): The nodes positions (N_nodes, 2)
        connectivity: The vertices indices (N_e, 3)
    Returns:
        elem_ids (torch.Tensor): The element indices found that correspond to the positions of the points (N_pts,)
    """
    vertices = nodes_positions[connectivity]  # (N_e, 3, 2)
    inside = is_in_triangle(x, vertices)  # (N_pts, N_e)

    # First valid element for each query point
    elem_ids = inside.long().argmax(dim=1)  # (N_pts,)

    # Detect points not in any element
    not_found = ~inside.any(dim=1)
    if not_found.any():
        missing = x[not_found]
        raise ValueError(f"No element found for points: '{missing}'")

    return elem_ids


class Mesh(nn.Module):
    """
    A mesh is defined by:
    * Its topoloy (nodes indices and nodes indices defining connectivity)
    * Its nodes' positions

    Args:
        topology (Topology): The mesh topology.
        nodes_positions (Field | TrainableField): A Field or TrainableField representing nodes positions.

    Attributes:
        topology (Topology): The mesh topology.
        nodes_positions (Field | TrainableField): A Field or TrainableField representing nodes positions.

    Raises:
        ValueError: If self.topology differs from nodes_positions.topology.
    """

    def __init__(self, topology, nodes_positions):
        super().__init__()

        self.topology = topology
        self.nodes_positions = nodes_positions
        self.dim = self.nodes_positions.dim

        if self.topology is not self.nodes_positions.topology:
            raise ValueError(
                f"Mesh self.topology does not correspond to self.nodes_positions.topology"
            )

    @property
    def n_nodes(self):
        return self.topology.n_nodes

    @property
    def n_elements(self):
        return self.topology.n_elements

    def elements_at(self, x):
        nodes = self.nodes_positions.full_values()  # (N_nodes, dim)
        connectivity = self.topology.connectivity  # (N_e, n_nodes_per_elem)

        if self.dim == 1:
            # (N_nodes,) -> element intervals
            x_nodes = nodes[connectivity]  # (N_e, 2)
            x_lo = x_nodes[:, 0]  # (N_e,)
            x_hi = x_nodes[:, 1]  # (N_e,)

            inside = (x[:, None] >= x_lo[None, :]) & (
                x[:, None] <= x_hi[None, :]
            )  # (N_pts, N_e)

            elem_ids = inside.long().argmax(dim=1)

            not_found = ~inside.any(dim=1)
            if not_found.any():
                raise ValueError(f"No element found for points: {x[not_found]}")

            return elem_ids

        elif self.dim == 2:
            return elements_at_2d(x.squeeze(), nodes, connectivity)
