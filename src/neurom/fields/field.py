import torch

from neurom.fields.field_base import FieldBase
from neurom.meshes.topology import Topology


class Field(FieldBase):
    """Interface of a Field

    A Field is defined at the nodal points (number of vertices) where the interpolation is performed, based on the topology. As such a Field tensor has shape (N_vertices, dim) where dim is the dimension of the field.
    The values of the field are registered as a buffer.

    Note:
        Compared to a TrainableField, a Field cannot be trained  (its values are fixed).

    Args:
        name (str): The Field's name.
        topology (Topology): The topology on which the Field is based.
        init_values (torch.Tensor): The initial values of the Field
        constraint (Constraint): The constraint which is imposed on the Field.

    Attributes:
        name (str): The Field's name.
        topology (Topology): The topology on which the Field is based.
        values (torch.nn.parameter.Buffer): The values at vertices.

    Raises:
         ValueError: If there is a different amount of nodes than there are field values.
         ValueError: If values does not provide field dimension, i.e. if it has a tensor shape of 1.
    """

    def __init__(
        self,
        name: str,
        topology: Topology,
        values,
    ):
        super().__init__(name=name, topology=topology)

        n_nodes = self.topology.n_nodes
        shape_values = values.shape
        if len(shape_values) <= 1:
            raise ValueError(
                f"Given 'values' has shape {shape_values}, but we expect it to be of shape (N_nodes, dim) with dim the field dimension."
            )

        n_values = shape_values[0]
        if n_values != n_nodes:
            raise ValueError(
                f"Given 'values' has a different number of values ({n_values}) than number of nodes in self.topology ({n_nodes})"
            )

        # Initialize reduced DOFs
        self.register_buffer("values", values)
        self.dim = values.shape[1]

    def full_values(self):
        """Get the full values

        Simply returns self.values
        """
        return self.values

    def at_elements(self):
        """Get the full values at elements

        Get the full values but per element following the topology connectivity.
        """
        return self.full_values()[self.topology.connectivity]
