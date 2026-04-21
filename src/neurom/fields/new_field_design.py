import torch
import torch.nn as nn

from neurom.constraints.constraint import Constraint
from neurom.fields.field_base import FieldBase
from neurom.meshes.topology import Topology


class FieldA(FieldBase):
    """Interface of a Field"""

    def __init__(
        self,
        name: str,
        topology: Topology,
        values,
        constraint: Constraint = NoConstraint(),
        trainable: bool = False,
    ):
        super().__init__(name=name, topology=topology)

        self.constraint = constraint

        # Ask constraint for free DOFs
        dofs_free = self.constraint.get_dofs_free(topology.n_nodes)
        self.register_buffer("dofs_free", dofs_free)

        # Initialize reduced DOFs
        self.values_reduced = nn.Parameter(values[dofs_free])

        self._trainable = trainable

    @property
    def dim(self):
        return self.values_reduced.shape[1]

    def freeze(self):
        self.values_reduced.requires_grad_(False)
        self._trainable = False

    def unfreeze(self):
        self.values_reduced.requires_grad_(True)
        self._trainable = True

    def is_trainable(self):
        return self._trainable

class FieldB(FieldBase):
    """Interface of a Field"""

    def __init__(
            self,
                name: str,
                topology: Topology,
                values,
                constraint: Constraint | None = None,
                trainable: bool = False,
            ):
            super().__init__(name=name, topology=topology)

            self.constraint = constraint
            self._trainable = trainable

            if self._trainable and not self.constraint:
                raise ValueError("A constraint must be provided if the field is trainable.")
            if not self._trainable and self.constraint:
                raise ValueError("A constraint should not be provided if the field is not trainable.")
                

            # Ask constraint for free DOFs
            dofs_free = self.constraint.get_dofs_free(topology.n_nodes)
            self.register_buffer("dofs_free", dofs_free)

            # Initialize reduced DOFs
            self.values_reduced = nn.Parameter(values[dofs_free])


        def freeze(self):
            self.values_reduced.requires_grad_(False)
            self._trainable = False

        def unfreeze(self):
            self.values_reduced.requires_grad_(True)
            self._trainable = True

        def is_trainable(self):
            return self._trainable
