import torch
import torch.nn as nn

from neurom.constraints import Constraint


class Field(nn.Module):
    def __init__(
        self,
        name,
        topology,
        init_values,
        constraint,
        trainable,
    ):
        super().__init__()

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
