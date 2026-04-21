"""Utilities for representing sampling points in neurom meshes.

This module defines abstract and concrete sampling classes that wrap a
``torch.Tensor`` containing coordinate data. The tensor shape encodes batch
dimensions (e.g., number of nodes, elements) followed by field dimensions
(e.g., spatial coordinates). The classes provide convenient properties for
accessing the shape, batch dimensions, and field dimensions.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch  # type: ignore


@dataclass
class Sampling(ABC):
    """Base class for different types of samplings.

    The values tensor can have different shapes depending on the specific sampling type (e.g., nodal, element, quadrature).

    Attributes:
        values (torch.Tensor): Tensor containing the sampling values.
    """

    values: torch.Tensor

    def __add__(self, o):
        if not isinstance(o, Sampling):
            return NotImplemented
        return self.__class__(self.values + o.values)

    def __sub__(self, o):
        if not isinstance(o, Sampling):
            return NotImplemented
        return self.__class__(self.values - o.values)

    def __neg__(self):
        return self.__class__(-self.values)

    def __mul__(self, s: float):
        breakpoint()
        return self.__class__(s * self.values)

    def __rmul__(self, s: float):
        breakpoint()
        return self.__class__(s * self.values)

    @property
    def shape(self) -> torch.Size:
        """Shape of the sampling tensor.

        Returns:
            torch.Size: Shape of ``values``.
        """
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the sampling tensor.

        Returns:
            int: ``values.ndim``.
        """
        return self.values.ndim

    @property
    @abstractmethod
    def batch_shape(self) -> torch.Size:
        """Shape of the batch dimensions.

        Returns:
            torch.Size: Shape representing the batch part of ``values``.
        """
        ...

    @property
    def f_shape(self) -> torch.Size:
        """Shape of the field dimensions (excluding batch dimensions).

        Returns:
            torch.Size: Shape of ``values`` after removing batch dimensions.
        """
        return torch.Size(self.values.shape[len(self.batch_shape) :])


@dataclass
class NodalSampling(Sampling):
    """Nodal sampling represents the coordinates of the nodes in a mesh.

    The values tensor has shape ``(n_nodes, dim)`` where ``n_nodes`` is the number of nodes and ``dim`` is the spatial dimension (e.g., 2 for 2D, 3 for 3D).
    """

    def __post_init__(self) -> None:
        """Validate that the sampling tensor has at least two dimensions (nodes, dim)."""
        assert self.ndim >= 2, (
            "Nodal sampling requires at least 2 dimensions (n_nodes, dim)"
        )

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the quadrature sampling.

        Returns:
            torch.Size: Size of the first dimension of ``values``.
        """
        return self.shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape for nodal sampling.

        Returns:
            torch.Size: First dimension of ``values`` representing nodes.
        """
        return self.shape[:1]


@dataclass
class ElementSampling(Sampling):
    """Element sampling represents the coordinates of the elements in a mesh.

    The values tensor has shape ``(n_elements, dim)`` where ``n_elements`` is the number of elements and ``dim`` is the spatial dimension (e.g., 2 for 2D, 3 for 3D).
    """

    def __post_init__(self) -> None:
        """Validate that the sampling tensor has at least two dimensions (elements, dim)."""
        assert self.ndim >= 2, (
            "Element sampling requires at least 2 dimensions (n_elements, dim)"
        )

    @property
    def n_elements(self) -> int:
        """Number of elements in the quadrature sampling.

        Returns:
            torch.Size: Size of the first dimension of ``values``.
        """
        return self.shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape for element sampling.

        Returns:
            torch.Size: First dimension of ``values`` representing elements.
        """
        return self.shape[:1]


@dataclass
class QuadratureSampling(Sampling):
    """Quadrature sampling represents the coordinates of the quadrature points in a mesh.

    The values tensor has shape ``(n_elements, n_quadrature, dim)`` where:

    - ``n_elements``: number of elements
    - ``n_quadrature``: number of quadrature points per element
    - ``dim``: spatial dimension (e.g., 2 for 2D, 3 for 3D).
    """

    def __post_init__(self) -> None:
        """Validate that the sampling tensor has at least three dimensions (elements, quadrature, dim)."""
        assert self.ndim >= 3, (
            "Quadrature sampling requires at least 3 dimensions (n_elements, n_quadrature, dim)"
        )

    @property
    def n_elements(self) -> int:
        """Number of elements in the quadrature sampling.

        Returns:
            torch.Size: Size of the first dimension of ``values``.
        """
        return self.shape[0]

    @property
    def n_quadrature(self) -> int:
        """Number of quadrature points per element.

        Returns:
            torch.Size: Size of the second dimension of ``values``.
        """
        return self.shape[1]

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape for quadrature sampling.

        Returns:
            torch.Size: Tuple of (n_elements, n_quadrature).
        """
        return self.shape[:2]
