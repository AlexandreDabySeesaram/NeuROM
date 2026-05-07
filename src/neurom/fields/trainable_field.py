import torch
import torch.nn as nn

from neurom.constraints.constraint import Constraint
from neurom.fields.field_base import FieldBase
from neurom.meshes.connectivity import Connectivity


class TrainableField(FieldBase):
    """Interface of a TrainableField

    A TrainableField is defined at the nodal points where the interpolation is performed, based on the connectivity. As such a TrainableField tensor has shape (N_vertices, dim) where dim is the dimension of the field.
    It owns a Constraint which defines a mask over indices of values which can actually vary (`values_reduced`) and the one that are imposed.

    Args:
        name (str): The TrainableField's name.
        connectivity (Connectivity): The connectivity on which the TrainableField is based.
        init_values (torch.Tensor): The initial values of the TrainableField at the nodes.
        constraint (Constraint): The constraint which is imposed on the TrainableField.

    Attributes:
        name (str): The TrainableField's name.
        connectivity (Connectivity): The connectivity on which the TrainableField is based.
        constraint (Constraint): The constraint which is imposed on the TrainableField.
        dofs_free (torch.Tensor): A boolean mask of nodes indices which is set to True if the DoF is free (can be trained) and False if the DoF has an imposed value (based on self.constraint).
        values_reduced (torch.nn.Parameter): The reduced values, i.e. the full values without the constrained ones.
    """

    def __init__(
        self,
        name: str,
        connectivity: Connectivity,
        init_values,
        constraint,
    ):
        super().__init__(name=name, connectivity=connectivity)

        self.constraint = constraint

        # Ask constraint for free DOFs
        dofs_free = self.constraint.get_dofs_free(connectivity.n_nodes)
        self.register_buffer("dofs_free", dofs_free)

        # Initialize reduced DOFs
        self.values_reduced = nn.Parameter(init_values[dofs_free])
        self.dim = init_values.shape[1]

    def full_values(self):
        """Get the full values

        Expand the reduced values over free dofs with the constrained ones.
        """
        return self.constraint.expand(self.values_reduced, self.dofs_free)

    def at_elements(self):
        """Get the full values at elements

        Get the full values but per element following the connectivity connectivity.
        """
        return self.full_values()[self.connectivity.element_connectivity]

    def freeze(self):
        self.values_reduced.requires_grad_(False)
