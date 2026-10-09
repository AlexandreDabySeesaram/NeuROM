"""Geometry utilities: isoparametric mappings and coordinate conversions."""

from neurom.geometry.iso_parametric_mapping_1d import IsoparametricMapping1D
from neurom.geometry.iso_parametric_mapping_2d import IsoparametricMapping2D
from neurom.geometry.barycentric_to_reference import barycentric_to_reference


def isoparametric_mapping(mesh):
    """Build the isoparametric mapping for ``mesh``, dispatching on its dimension.

    The geometry basis is taken from the mesh's coordinate element, so no shape function is
    passed in.

    Args:
        mesh (Mesh): The mesh to map.

    Returns:
        IsoparametricMapping1D | IsoparametricMapping2D: The mapping for ``mesh.dim``.

    Raises:
        NotImplementedError: If ``mesh.dim`` is neither 1 nor 2.
    """
    if mesh.dim == 1:
        return IsoparametricMapping1D(mesh)
    if mesh.dim == 2:
        return IsoparametricMapping2D(mesh)
    raise NotImplementedError(
        f"No isoparametric mapping for dim={mesh.dim} (only 1-D and 2-D)."
    )


__all__ = [
    "IsoparametricMapping1D",
    "IsoparametricMapping2D",
    "isoparametric_mapping",
    "barycentric_to_reference",
]
