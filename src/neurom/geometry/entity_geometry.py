r"""Per-facet entity geometry derived from a mesh -- the entity-level sibling of the cell mapping.

:class:`~neurom.geometry.affine_simplex_mapping.AffineSimplexMapping` provides *cell* geometry (the
Jacobian :math:`J_e`); :class:`EntityGeometry` provides *facet* geometry/topology that a
direction-carrying (H(div)/H(curl)) element needs: the orientation sign :math:`\sigma_{e,s}` now,
and -- with the flux functional (WP9 stage C) -- the physical normal :math:`\mathbf n_E` and measure
:math:`|E|`.

Dimension-generic: everything is expressed on the mesh's **facet** dimension
(:attr:`~neurom.meshes.Topology.facet_dim`) -- edges in 2-D, faces in 3-D -- rather than hard-coded
to edges, so it extends to 3-D once the topology enumerates faces (the normal/measure will use the
dimension-independent Nanson form, not the 2-D rotate-the-tangent trick).
"""

import torch
import torch.nn as nn

from neurom.meshes.mesh import Mesh


class EntityGeometry(nn.Module):
    r"""Per-(cell, local-facet) orientation signs (and, later, normals/measures) for an element.

    Args:
        mesh (Mesh): The mesh whose facet geometry this describes.

    Attributes:
        facet_dim (int): The facet dimension (``topology.facet_dim``) this object works on.
    """

    def __init__(self, mesh: Mesh):
        super().__init__()
        self._mesh = mesh
        self.facet_dim = mesh.topology.facet_dim
        # -90 deg rotation: n = R t is the OUTWARD normal on a CCW cell (plan 3.1). As a row op,
        # (t0, t1) @ R.T = (t1, -t0).
        self._R = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def _edge_normal_length(self):
        r"""Per **global** edge: unit canonical normal :math:`\hat{\mathbf n}=R\hat{\mathbf t}` and length.

        Canonical (from ``edge_vertices`` ``lo -> hi``) so the two cells sharing an edge see the SAME
        normal -- the per-cell local/global flip is the orientation sign (T_e), handled elsewhere.
        In-graph in the node positions (r-adaptivity). **2-D only** (the ``R`` rotation); 3-D faces
        would use the cross-product / Nanson form ``n ∝ det(J) J^{-T} n_ref`` once faces are built.
        """
        x = self._mesh.coordinates.full_values()  # (n_vert, gdim)
        ev = self._mesh.topology.edge_vertices  # (n_edges, 2) canonical (lo, hi)
        t = x[ev[:, 1]] - x[ev[:, 0]]  # (n_edges, gdim) tangent lo -> hi
        length = t.norm(dim=-1, keepdim=True)  # (n_edges, 1) |E|
        normal = (t @ self._R.to(t.dtype).T) / length  # (n_edges, gdim) unit n = R t / |E|
        return normal, length

    def facet_normals(self, element_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""Outward unit facet normal :math:`\hat{\mathbf n}_E` per cell/local facet, ``(n_e, n_local, gdim)``."""
        normal, _ = self._edge_normal_length()
        ce = self._mesh.topology.cell_entities(self.facet_dim)
        ce = ce if element_ids is None else ce[element_ids]
        return normal[ce]

    def facet_measures(self, element_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""Facet measure :math:`|E|` (edge length in 2-D) per cell/local facet, ``(n_e, n_local, 1)``."""
        _, length = self._edge_normal_length()
        ce = self._mesh.topology.cell_entities(self.facet_dim)
        ce = ce if element_ids is None else ce[element_ids]
        return length[ce]

    def facet_signs(self, element_ids: torch.Tensor | None = None) -> torch.Tensor:
        r"""Orientation sign :math:`\sigma_{e,s} \in \{+1, -1\}` per cell and local facet.

        ``+1`` where the cell's local facet is in canonical (sorted-vertex) order, ``-1`` otherwise --
        the sign two cells sharing a facet must agree on for a direction-carrying DOF (an RT0 flux),
        i.e. the diagonal of :math:`T_e`. Purely topological (no node positions).

        Args:
            element_ids (torch.Tensor | None): Element subset ``(n_e,)``, or ``None`` for all.

        Returns:
            torch.Tensor: ``(n_e, n_local_facets)`` of :math:`\pm 1`.
        """
        sign = self._mesh.topology.entity_orientation(self.facet_dim)
        return sign if element_ids is None else sign[element_ids]
