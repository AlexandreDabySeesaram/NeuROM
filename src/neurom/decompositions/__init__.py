"""Tensor decompositions for a-priori model order reduction."""

from neurom.decompositions.tensor_decomposition import TensorDecomposition
from neurom.decompositions.factor import FactorSpace, MonomSpec
from neurom.decompositions.pgd import CPPGD

__all__ = [
    "TensorDecomposition",
    "FactorSpace",
    "MonomSpec",
    "CPPGD",
]
