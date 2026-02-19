import torch
import torch.nn as nn

from neurom.constraints.constraint import Constraint
from neurom.meshes import Topology


class Field(nn.Module):
    """Interface of a Field

    A Field is defined at the nodal points where the interpolation is performed, based on the topology.
    It owns a Constraint which defines a mask over indices of values which can actually vary (`values_reduced`) and the one that are imposed. The Field can be set to be trainable or not.
    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        name (str): The Field's name.
        topology (Topology): The topology on which the Field is based.
        init_values (torch.Tensor): The initial values of the Field
        constraint (Constraint): The constraint which is imposed on the Field.
        is_trainable (bool): Whether the Field is trainable or not.

    Attributes:
        name (str): The Field's name.
        topology (Topology): The topology on which the Field is based.
        constraint (Constraint): The constraint which is imposed on the Field.
        dofs_free (torch.Tensor): A boolean mask of nodes indices which is set to True if the DoF is free (can be trained) and False if the DoF has an imposed value (based on self.constraint).
        values_reduced (torch.nn.Parameter | torch.nn.parameter.Buffer): The reduced values, i.e. the full values without the constrained ones.
    """

    def __init__(
        self,
        name: str,
        topology: Topology,
        init_values,
        constraint,
        trainable,
    ):
        super().__init__()

        self.name = name
        self.topology = topology
        self.constraint = constraint

        # Ask constraint for free DOFs
        dofs_free = self.constraint.get_dofs_free(topology.n_nodes)
        self.register_buffer("dofs_free", dofs_free)

        # Initialize reduced DOFs
        reduced_init = init_values[dofs_free]
        if trainable:
            self.values_reduced = nn.Parameter(reduced_init)
        else:
            self.register_buffer("values_reduced", reduced_init)

    def full_values(self):
        return self.constraint.expand(self.values_reduced, self.dofs_free)

    def at_elements(self):
        return self.full_values()[self.topology.conn]
