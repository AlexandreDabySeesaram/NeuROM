"""Dirichlet boundary-condition constraint for trainable fields."""

import torch


from neurom.constraints.constraint import Constraint


class Dirichlet(Constraint):
    """Constraint that imposes fixed Dirichlet values on a set of nodes.

    A ``Dirichlet`` constraint marks a list of node indices as constrained and
    stores the value that each constrained node must take.  The remaining nodes
    are considered free and their values are learned during training.

    Attributes:
        nodes (torch.Tensor): 1-D integer tensor of shape ``(n_constrained,)``
            containing the indices of the constrained nodes.  Registered as a
            buffer.
        values_imposed (torch.Tensor): Tensor of shape
            ``(n_constrained, dim)`` containing the imposed field values for
            the constrained nodes.  Registered as a buffer.
    """

    def __init__(self, nodes, values_imposed):
        """Initialize the Dirichlet constraint.

        Args:
            nodes (array-like or torch.Tensor): Indices of the nodes that have
                an imposed (Dirichlet) boundary condition.
            values_imposed (array-like or torch.Tensor): Imposed field values
                for those nodes, of shape ``(n_constrained, dim)``.
        """
        super().__init__()
        self.register_buffer("nodes", torch.as_tensor(nodes))
        self.register_buffer("values_imposed", torch.as_tensor(values_imposed))

    def get_dofs_free(self, n_nodes):
        """Return a boolean mask identifying which DOFs are free.

        Constructs a boolean tensor of length ``n_nodes`` initialised to
        ``True``, then sets the entries at ``self.nodes`` to ``False``.

        Args:
            n_nodes (int): Total number of nodes in the mesh.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(n_nodes,)`` where
            ``True`` indicates a free DOF and ``False`` indicates a
            Dirichlet-constrained DOF.
        """
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        dofs_free[self.nodes] = False
        return dofs_free

    def expand(self, reduced_values, dofs_free):
        """Reconstruct full nodal values from free-DOF and imposed values.

        Allocates a zero tensor of shape ``(n_nodes, dim)`` and fills the
        free-DOF slots with ``reduced_values`` and the constrained-DOF slots
        with ``self.values_imposed``.

        Args:
            reduced_values (torch.Tensor): Values for the free DOFs only, of
                shape ``(n_free, dim)``.
            dofs_free (torch.Tensor): Boolean mask of shape ``(n_nodes,)``
                where ``True`` marks a free DOF.

        Returns:
            torch.Tensor: Full nodal value tensor of shape ``(n_nodes, dim)``
            assembled from ``reduced_values`` and ``self.values_imposed``.
        """
        full = torch.zeros(
            reduced_values.shape[0] + self.values_imposed.shape[0],
            reduced_values.shape[1],
            device=reduced_values.device,
            dtype=reduced_values.dtype,
        )

        full[dofs_free, :] = reduced_values
        full[~dofs_free, :] = self.values_imposed
        return full
