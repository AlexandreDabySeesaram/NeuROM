import torch

from neurom.constraints.constraint import Constraint


class NoConstraint(Constraint):
    """Case where is no constraint."""

    def get_dofs_free(self, n_nodes):
        """
        Return boolean mask of free DOFs (True = free, False = constrained)

        Return a tensor of shape n_nodes with all dofs set to True.
        """
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        return dofs_free

    def expand(self, reduced_values, dofs_free, **kwargs):
        """
        Expand the reduced values with the imposed ones

        Does nothing, returns a tensor with only the reduced values.

        Returns:
            The fully assembled values with only reduces values.
        """
        # full vector
        full = torch.zeros(
            reduced_values.shape,
            device=reduced_values.device,
            dtype=reduced_values.dtype,
        )
        full[dofs_free] = reduced_values
        return full
