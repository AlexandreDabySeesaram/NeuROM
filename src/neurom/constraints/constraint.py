from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Constraint(nn.Module, ABC):
    """
    A base class representing a constraint to a Field.

    A Constraint provides ways to constraint the values of a Field.
    It provides a expand() method which reconstruct the full values over the full fields based on the constraints and the free values.
    and a get_dofs_free() which returns a boolean `dofs_free` tensor which represent a mask: if True, the dof is free and if False, it has an imposed value.
    """

    @abstractmethod
    def expand(self, reduced_values):
        """Expand the reduced values with the imposed ones"""
        pass

    @abstractmethod
    def get_dofs_free(self, n_nodes):
        """Return boolean mask of free DOFs (True = free, False = constrained)"""
