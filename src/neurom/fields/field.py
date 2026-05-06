import torch

from neurom.fields.field_base import FieldBase
from neurom.meshes import Topology


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
    """

    def __init__(
        self,
        name: str,
        topology: Topology,
        values,
    ):
        super().__init__(name=name, topology=topology)

        # Initialize reduced DOFs
        self.register_buffer("values", values)

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
