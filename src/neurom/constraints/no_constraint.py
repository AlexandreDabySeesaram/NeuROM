"""Identity constraint that leaves all DOFs free."""

import torch

from neurom.constraints.constraint import Constraint


class NoConstraint(Constraint):
    """Constraint that imposes no restrictions — all DOFs are free.

    ``NoConstraint`` is used when every node in the mesh is free to vary
    during training.  :meth:`get_dofs_free` returns an all-``True`` mask and
    :meth:`expand` simply copies the reduced values into the output tensor.
    """

    def get_dofs_free(self, n_nodes):
        """Return a boolean mask with all DOFs marked as free.

        Args:
            n_nodes (int): Total number of nodes in the mesh.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(n_nodes,)`` filled
            entirely with ``True``.
        """
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        return dofs_free

    def expand(self, reduced_values, dofs_free, **kwargs):
        """Return the full nodal values, which equal the reduced values.

        Since no DOFs are constrained, ``reduced_values`` already covers all
        nodes.  The method allocates a zero tensor and writes ``reduced_values``
        into the positions indicated by ``dofs_free`` (all positions).

        Args:
            reduced_values (torch.Tensor): Values for the free DOFs, of shape
                ``(n_nodes, dim)``.
            dofs_free (torch.Tensor): Boolean mask of shape ``(n_nodes,)``
                where all entries are ``True`` for this constraint.
            **kwargs: Accepted but ignored for API compatibility.

        Returns:
            torch.Tensor: Full nodal value tensor of shape ``(n_nodes, dim)``
            equal to ``reduced_values``.
        """
        # full vector
        full = torch.zeros(
            reduced_values.shape,
            device=reduced_values.device,
            dtype=reduced_values.dtype,
        )
        full[dofs_free] = reduced_values
        return full
