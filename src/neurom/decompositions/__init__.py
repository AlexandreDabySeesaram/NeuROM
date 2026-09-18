"""Tensor decompositions for a-priori model order reduction."""

from neurom.decompositions.tensor_decomposition import TensorDecomposition
from neurom.decompositions.pgd import Axis, CPPGD

__all__ = [
    "TensorDecomposition",
    "Axis",
    "CPPGD",
]
