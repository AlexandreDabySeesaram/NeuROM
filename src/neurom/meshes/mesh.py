"""Mesh data structure and point-location utilities."""

import torch.nn as nn


def is_in_triangle(pts, vertices):
    """Find if points are inside a triangle defined by its vertices.

    Args:
        pts (torch.Tensor): The points to check, shape ``(N_pts, 2)``.
        vertices (torch.Tensor): The triangle vertices, shape ``(N_e, 3, 2)``
            where the second dimension indexes the three vertices ``a``,
            ``b``, ``c`` and the last dimension holds the 2-D coordinates.

    Returns:
        torch.Tensor: Boolean tensor of shape ``(N_pts, N_e)`` indicating
        whether each point lies inside each triangle.
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


def elements_at_1d(x, nodes_positions, connectivity):
    """Find the 1-D element indices containing each query point.

    Args:
        x (torch.Tensor): The query points, shape ``(N_pts, 1)``.
        nodes_positions (torch.Tensor): The node positions, shape
            ``(N_nodes, 1)``.
        connectivity (torch.Tensor): The element connectivity (node indices
            per element), shape ``(N_e, 2)``.

    Returns:
        torch.Tensor: Element indices of shape ``(N_pts,)`` giving the index
        of the element that contains each query point.

    Raises:
        ValueError: If one or more query points do not lie in any element.
    """
    # (N_nodes,) -> element intervals
    x_nodes = nodes_positions[connectivity]  # (N_e, 2, 1)
    x_lo = x_nodes[:, 0]  # (N_e,)
    x_hi = x_nodes[:, 1]  # (N_e,)

    inside = (x[:, None] >= x_lo[None, :]) & (
        x[:, None] <= x_hi[None, :]
    )  # (N_pts, N_e)

    elem_ids = inside.long().argmax(dim=1)

    not_found = ~inside.any(dim=1)
    if not_found.any():
        raise ValueError(f"No element found for points: {x[not_found]}")

    return elem_ids.squeeze(-1)


def elements_at_2d(x, nodes_positions, connectivity):
    """Find the 2-D element indices containing each query point.

    Args:
        x (torch.Tensor): The query points, shape ``(N_pts, 2)``.
        nodes_positions (torch.Tensor): The node positions, shape
            ``(N_nodes, 2)``.
        connectivity (torch.Tensor): The element connectivity (node indices
            per element), shape ``(N_e, 3)``.

    Returns:
        torch.Tensor: Element indices of shape ``(N_pts,)`` giving the index
        of the element that contains each query point.

    Raises:
        ValueError: If one or more query points do not lie in any element.
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
    """A finite-element mesh combining topology and node positions.

    A mesh is defined by its topology (node indices and element connectivity)
    together with the spatial positions of each node.

    Args:
        connectivity (Connectivity): The mesh connectivity (node indices and
            element-to-node mapping).
        nodes_positions (Field or TrainableField): A ``Field`` or
            ``TrainableField`` holding the spatial coordinates of every node.

    Attributes:
        connectivity (Connectivity): The mesh connectivity.
        nodes_positions (Field or TrainableField): The spatial coordinates of
            every node.
        dim (int): Spatial dimension of the mesh, taken from
            ``nodes_positions.dim``.

    Raises:
        ValueError: If ``connectivity`` is not the same object as
            ``nodes_positions.connectivity``.
    """

    def __init__(self, connectivity, nodes_positions):
        super().__init__()

        self.connectivity = connectivity
        self.nodes_positions = nodes_positions
        self.dim = self.nodes_positions.dim

        if self.connectivity is not self.nodes_positions.connectivity:
            raise ValueError(
                "Mesh self.connectivity does not correspond to self.nodes_positions.connectivity"
            )

    @property
    def n_nodes(self):
        """The number of nodes in the mesh.

        Returns:
            int: The number of nodes.
        """
        return self.connectivity.n_nodes

    @property
    def n_elements(self):
        """The number of elements in the mesh.

        Returns:
            int: The number of elements.
        """
        return self.connectivity.n_elements

    def elements_at(self, x):
        """Find the element index containing each query point.

        Dispatches to :func:`elements_at_1d` or :func:`elements_at_2d`
        depending on ``self.dim``.

        Args:
            x (torch.Tensor): The query points, shape ``(N_pts, dim)`` or
                broadcastable to it.

        Returns:
            torch.Tensor: Element indices of shape ``(N_pts,)`` giving the
            index of the element that contains each query point.

        Raises:
            ValueError: If one or more query points do not lie in any element.
        """
        nodes = self.nodes_positions.full_values()  # (N_nodes, dim)
        connectivity = self.connectivity.element_connectivity  # (N_e, n_nodes_per_elem)

        if self.dim == 1:
            return elements_at_1d(x.squeeze().unsqueeze(-1), nodes, connectivity)
        elif self.dim == 2:
            return elements_at_2d(x.squeeze(), nodes, connectivity)
