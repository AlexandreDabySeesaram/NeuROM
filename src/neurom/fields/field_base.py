"""Abstract base class for nodal fields in neurom FEM models."""

from abc import ABC, abstractmethod
import torch.nn as nn

from neurom.meshes.connectivity import Connectivity


class FieldBase(nn.Module, ABC):
    """Abstract base class for all nodal field types.

    A field is defined at the nodes of a mesh and provides methods to retrieve
    the full nodal values and the values gathered per element.  Concrete
    subclasses must implement :meth:`full_values` and :meth:`at_elements`.

    Attributes:
        name (str): Human-readable identifier for the field.
        connectivity (Connectivity): Mesh connectivity that defines the node
            indices and element-to-node mapping used by the field.
    """

    def __init__(
        self,
        name: str,
        connectivity: Connectivity,
    ):
        """Initialize the field base with a name and connectivity.

        Args:
            name (str): Human-readable identifier for the field.
            connectivity (Connectivity): Mesh connectivity that describes the
                node indices and element-to-node mapping.
        """
        super().__init__()

        self.name = name
        self.connectivity = connectivity

    @abstractmethod
    def full_values(self):
        """Return the complete nodal values across all degrees of freedom.

        Concrete subclasses expand the (possibly reduced) stored values so that
        both free and constrained degrees of freedom are represented.

        Returns:
            torch.Tensor: Nodal field values of shape ``(n_nodes, dim)``.
        """
        pass

    @abstractmethod
    def at_elements(self):
        """Return nodal values gathered per element.

        Uses the element connectivity to index into the full nodal values,
        producing one value block per element.

        Returns:
            torch.Tensor: Field values indexed by element connectivity, of
            shape ``(n_elements, n_simplex, dim)``.
        """
        pass
