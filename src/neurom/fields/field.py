"""Non-trainable nodal field with fixed values stored as a buffer."""

from neurom.fields.field_base import FieldBase
from neurom.meshes.connectivity import Connectivity


class Field(FieldBase):
    """A nodal field with fixed (non-trainable) values.

    A ``Field`` stores values at every node of the mesh.  The values tensor
    has shape ``(n_nodes, dim)`` where ``dim`` is the field dimension.  Values
    are registered as an ``nn.Module`` buffer and therefore cannot be updated
    by an optimizer.

    Note:
        To create a field whose values can be updated during training, use
        :class:`~neurom.fields.TrainableField` instead.

    Attributes:
        name (str): Human-readable identifier for the field.
        connectivity (Connectivity): Mesh connectivity that defines the
            element-to-node mapping used by the field.
        values (torch.Tensor): Fixed nodal values of shape
            ``(n_nodes, dim)``, registered as a non-trainable buffer.

    Raises:
        ValueError: If ``values`` has fewer than two dimensions, i.e. if the
            field dimension ``dim`` is not provided.
        ValueError: If the number of rows in ``values`` does not match the
            number of nodes in ``connectivity``.
    """

    def __init__(
        self,
        name: str,
        connectivity: Connectivity,
        values,
    ):
        """Initialize a fixed nodal field.

        Args:
            name (str): Human-readable identifier for the field.
            connectivity (Connectivity): Mesh connectivity that defines the
                element-to-node mapping.
            values (torch.Tensor): Initial nodal values of shape
                ``(n_nodes, dim)``.  Must have at least two dimensions.

        Raises:
            ValueError: If ``values`` has fewer than two dimensions.
            ValueError: If ``values.shape[0]`` differs from the number of
                nodes in ``connectivity``.
        """
        super().__init__(name=name, connectivity=connectivity)

        n_nodes = self.connectivity.n_nodes
        shape_values = values.shape
        if len(shape_values) <= 1:
            raise ValueError(
                f"Given 'values' has shape {shape_values}, but we expect it to be of shape (N_nodes, dim) with dim the field dimension."
            )

        n_values = shape_values[0]
        if n_values != n_nodes:
            raise ValueError(
                f"Given 'values' has a different number of values ({n_values}) than number of nodes in self.connectivity ({n_nodes})"
            )

        # Initialize reduced DOFs
        self.register_buffer("values", values)

    @property
    def dim(self):
        """Field dimension (number of components per node).

        Returns:
            int: Size of the second dimension of ``self.values``.
        """
        return self.values.shape[1]

    def full_values(self):
        """Return the complete nodal values.

        For a fixed ``Field`` no expansion is necessary; the stored buffer is
        returned directly.

        Returns:
            torch.Tensor: Nodal field values of shape ``(n_nodes, dim)``.
        """
        return self.values

    def at_elements(self):
        """Return nodal values gathered per element.

        Indexes :meth:`full_values` with the element connectivity to produce
        one block of nodal values per element.

        Returns:
            torch.Tensor: Field values of shape
            ``(n_elements, n_simplex, dim)``.
        """
        return self.full_values()[self.connectivity.element_connectivity]
