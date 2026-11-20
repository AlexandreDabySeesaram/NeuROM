"""Geometry utilities: the affine reference-to-physical mapping and coordinate conversions."""

from neurom.geometry.affine_simplex_mapping import AffineSimplexMapping
from neurom.geometry.entity_geometry import EntityGeometry
from neurom.geometry.barycentric_to_reference import barycentric_to_reference


__all__ = [
    "AffineSimplexMapping",
    "EntityGeometry",
    "barycentric_to_reference",
]
