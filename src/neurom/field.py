import torch
import torch.nn as nn

from neurom.constraints.constraint import Constraint
from neurom.meshes import Topology


class FieldBase(nn.Module):
    def __init__(
        self,
        name: str,
        topology: Topology,
    ):
        super().__init__()

        self.name = name
        self.topology = topology


class TrainableField(FieldBase):
    """Interface of a TrainableField

    A TrainableField is defined at the nodal points where the interpolation is performed, based on the topology.
    It owns a Constraint which defines a mask over indices of values which can actually vary (`values_reduced`) and the one that are imposed.

    Args:
        name (str): The TrainableField's name.
        topology (Topology): The topology on which the TrainableField is based.
        init_values (torch.Tensor): The initial values of the TrainableField
        constraint (Constraint): The constraint which is imposed on the TrainableField.

    Attributes:
        name (str): The TrainableField's name.
        topology (Topology): The topology on which the TrainableField is based.
        constraint (Constraint): The constraint which is imposed on the TrainableField.
        dofs_free (torch.Tensor): A boolean mask of nodes indices which is set to True if the DoF is free (can be trained) and False if the DoF has an imposed value (based on self.constraint).
        values_reduced (torch.nn.Parameter): The reduced values, i.e. the full values without the constrained ones.
    """

    def __init__(
        self,
        name: str,
        topology: Topology,
        init_values,
        constraint,
    ):
        super().__init__(name=name, topology=topology)

        self.constraint = constraint

        # Ask constraint for free DOFs
        dofs_free = self.constraint.get_dofs_free(topology.n_nodes)
        self.register_buffer("dofs_free", dofs_free)

        # Initialize reduced DOFs
        self.values_reduced = nn.Parameter(init_values[dofs_free])

    def full_values(self):
        """Get the full values

        Expand the reduced values over free dofs with the constraine ones.
        """
        return self.constraint.expand(self.values_reduced, self.dofs_free)

    def at_elements(self):
        """Get the full values at elements

        Get the full values but per element following the topology connectivity.
        """
        return self.full_values()[self.topology.conn]


class Field(FieldBase):
    """Interface of a Field

    A Field is defined at the nodal points where the interpolation is performed, based on the topology.
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
        values (torch.nn.parameter.Buffer): The reduced values, i.e. the full values without the constrained ones.
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
        return self.values

    def at_elements(self):
        return self.full_values()[self.topology.conn]
