"""Abstract base class for constraints applied to trainable fields."""

from abc import ABC, abstractmethod
import torch.nn as nn


class Constraint(nn.Module, ABC):
    """Abstract base class for constraints applied to a ``TrainableField``.

    A ``Constraint`` partitions the nodes of a mesh into *free* DOFs (whose
    values are learned during training) and *constrained* DOFs (whose values
    are imposed externally).  Concrete subclasses must implement:

    - :meth:`get_dofs_free`: returns a boolean mask identifying free DOFs.
    - :meth:`expand`: reconstructs the full nodal value tensor from the
      reduced (free-DOF) values and the imposed values.
    """

    @abstractmethod
    def expand(self, reduced_values, dofs_free):
        """Reconstruct full nodal values from free-DOF and imposed values.

        Args:
            reduced_values (torch.Tensor): Values for the free DOFs only, of
                shape ``(n_free, dim)``.
            dofs_free (torch.Tensor): Boolean mask of shape ``(n_nodes,)``
                where ``True`` marks a free DOF.

        Returns:
            torch.Tensor: Full nodal value tensor of shape ``(n_nodes, dim)``
            with free-DOF entries taken from ``reduced_values`` and
            constrained-DOF entries filled with imposed values.
        """
        pass

    @abstractmethod
    def get_dofs_free(self, n_nodes):
        """Return a boolean mask identifying which DOFs are free.

        Args:
            n_nodes (int): Total number of nodes in the mesh.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(n_nodes,)`` where
            ``True`` indicates a free DOF and ``False`` indicates a
            constrained (imposed) DOF.
        """
