"""Mesh data structure and point-location utilities."""

import torch.nn as nn


def is_in_triangle(pts, vertices):
    """Find if points are inside a triangle defined by its vertices.

    Args:
        pts (torch.Tensor): The points to check, shape ``(N_pts, 2)``.
        vertices (torch.Tensor): The triangle vertices, shape ``(N_e, 3, 2)``
            where the second dimension indexes the three vertices ``a``,
            ``b``, ``c`` and the last dimension holds the 2-D coordinates.

    The test accepts either vertex winding: a point is inside when the three
    cross products share a sign, whichever it is. Points on an edge give a
    zero cross product and count as inside.

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

    # Sign, not orientation: the three cross products are all negative for a
    # triangle listed one way round and all positive for the other, so testing
    # a single sign silently rejects every point of a mesh wound the other way
    # -- and the caller then blames the point ("No element found"). Nothing in
    # the API asks for a particular winding, so accept both.
    same_sign = ((d0 <= 0) & (d1 <= 0) & (d2 <= 0)) | (
        (d0 >= 0) & (d1 >= 0) & (d2 >= 0)
    )
    return same_sign  # (N_pts, N_e)


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
    # Drop the trailing coordinate axis. A 1-D point is stored as (N_pts, 1),
    # but the point-by-element cross product below wants a plain (N_pts, N_e)
    # matrix -- the shape elements_at_2d already works with. Keeping the axis
    # makes `inside` (N_pts, N_e, 1) and forces a squeeze on the way out, an
    # asymmetry between the two implementations with nothing to justify it.
    x_flat = x[:, 0]  # (N_pts,)
    x_nodes = nodes_positions[connectivity]  # (N_e, 2, 1)
    x_lo = x_nodes[:, 0, 0]  # (N_e,)
    x_hi = x_nodes[:, 1, 0]  # (N_e,)

    inside = (x_flat[:, None] >= x_lo[None, :]) & (
        x_flat[:, None] <= x_hi[None, :]
    )  # (N_pts, N_e)

    # A point sitting exactly on a shared node is inside both elements;
    # argmax keeps the first one.
    elem_ids = inside.long().argmax(dim=1)  # (N_pts,)

    not_found = ~inside.any(dim=1)  # (N_pts,)
    if not_found.any():
        raise ValueError(f"No element found for points: {x[not_found]}")

    return elem_ids


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
            x (torch.Tensor): The query points, shape ``(N_pts, dim)``.

        Returns:
            torch.Tensor: Element indices of shape ``(N_pts,)`` giving the
            index of the element that contains each query point.

        Raises:
            ValueError: If ``x`` is not of shape ``(N_pts, dim)``, or if one or
                more query points do not lie in any element.
            NotImplementedError: If the mesh dimension is neither 1 nor 2.
        """
        # A point-wise query has exactly one shape: one row per point, one
        # column per spatial coordinate. Anything else has to be rejected here
        # rather than left to fail downstream, because it usually does not fail
        # at all -- it broadcasts. A flat (N_pts,) tensor pairs up against the
        # element axis instead of the point axis, and when N_pts happens to
        # equal N_e the result comes back the wrong length with the wrong
        # values and no error: four points on a four-element mesh return a
        # single index. The same trap bit PointWiseInterpolator.at_position,
        # where the shape functions sliced the broadcast product back down to
        # the *correct output shape* while the numbers were wrong.
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(
                f"elements_at expects x of shape (N_pts, dim) with "
                f"dim={self.dim}, got {tuple(x.shape)}. Reshape a flat "
                f"list of points with x.reshape(-1, {self.dim})."
            )

        nodes = self.nodes_positions.full_values()  # (N_nodes, dim)
        connectivity = self.connectivity.element_connectivity  # (N_e, n_nodes_per_elem)

        if self.dim == 1:
            return elements_at_1d(x, nodes, connectivity)
        elif self.dim == 2:
            return elements_at_2d(x, nodes, connectivity)
        else:
            # Without this branch the method falls off the end and returns
            # None, which does not raise where it is used: `tensor[None, :]`
            # is valid indexing that inserts an axis. The failure would then
            # surface several lines later as an unrelated shape error.
            raise NotImplementedError(
                f"Point location is only implemented for 1-D and 2-D meshes, "
                f"got dim={self.dim}."
            )
