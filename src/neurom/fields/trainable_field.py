"""Trainable nodal field with support for Dirichlet-type constraints."""

import torch.nn as nn

from neurom.fields.field_base import FieldBase
from neurom.meshes.connectivity import Connectivity


class TrainableField(FieldBase):
    """A nodal field whose free-DOF values are trainable parameters.

    A ``TrainableField`` stores values at every node of the mesh.  The values
    tensor has conceptual shape ``(n_nodes, dim)``.  A
    :class:`~neurom.constraints.Constraint` partitions the nodes into *free*
    DOFs (whose values are learned) and *constrained* DOFs (whose values are
    imposed).  Only the free values are stored as ``nn.Parameter``
    (``values_reduced``); the full nodal tensor is reconstructed on the fly by
    :meth:`full_values`.

    Attributes:
        name (str): Human-readable identifier for the field.
        connectivity (Connectivity): Mesh connectivity that defines the
            element-to-node mapping used by the field.
        constraint (Constraint): Constraint object that determines which DOFs
            are free and supplies imposed values for the constrained ones.
        dofs_free (torch.Tensor): Boolean mask of shape ``(n_nodes,)``; entry
            is ``True`` if the corresponding DOF is free (trainable) and
            ``False`` if it is constrained (imposed value).
        values_reduced (torch.nn.Parameter): Trainable parameter tensor
            containing the field values for the free DOFs only, of shape
            ``(n_free, dim)``.
        dim (int): Field dimension (number of components per node).
    """

    def __init__(
        self,
        name: str,
        connectivity: Connectivity,
        init_values,
        constraint,
    ):
        """Initialize a trainable nodal field.

        Args:
            name (str): Human-readable identifier for the field.
            connectivity (Connectivity): Mesh connectivity that defines the
                element-to-node mapping.
            init_values (torch.Tensor): Initial nodal values of shape
                ``(n_nodes, dim)``.  The free-DOF slice is used to initialise
                ``values_reduced``.
            constraint (Constraint): Constraint that defines the free/imposed
                DOF split and provides the imposed values during expansion.
        """
        super().__init__(name=name, connectivity=connectivity)

        self.constraint = constraint

        # Ask constraint for free DOFs
        dofs_free = self.constraint.get_dofs_free(connectivity.n_nodes)
        self.register_buffer("dofs_free", dofs_free)

        # Initialize reduced DOFs
        self.values_reduced = nn.Parameter(init_values[dofs_free])
        self.dim = init_values.shape[1]

    def full_values(self):
        """Return the complete nodal values by expanding the free-DOF parameter.

        Delegates to the constraint's ``expand`` method, which fills the free
        DOF slots with ``values_reduced`` and the constrained slots with the
        imposed values.

        Returns:
            torch.Tensor: Full nodal field values of shape ``(n_nodes, dim)``.
        """
        return self.constraint.expand(self.values_reduced, self.dofs_free)

    def at_elements(self):
        """Return nodal values gathered per element.

        Indexes :meth:`full_values` with the element connectivity to produce
        one block of nodal values per element.

        Returns:
            torch.Tensor: Field values of shape
            ``(n_elements, n_simplex, dim)``.
        """
        return self.full_values()[self.connectivity.element_connectivity]

    def freeze(self):
        """Disable gradient computation for the trainable DOF values.

        After calling this method, ``values_reduced`` will no longer
        accumulate gradients and the field will not be updated by an optimizer.
        """
        self.values_reduced.requires_grad_(False)
