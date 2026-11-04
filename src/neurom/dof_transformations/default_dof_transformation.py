"""Choice of the DOF transformation matching a shape function."""

from neurom.dof_transformations.dof_transformation import DofTransformation
from neurom.dof_transformations.identity_dof_transformation import (
    IdentityDofTransformation,
)
from neurom.dof_transformations.nodal_derivative_dof_transformation import (
    NodalDerivativeDofTransformation,
)
from neurom.shape_functions.shape_function import ShapeFunction


def default_dof_transformation(sf: ShapeFunction, mapping) -> DofTransformation:
    """Build the DOF transformation required by a shape function.

    Args:
        sf (ShapeFunction): Shape function whose ``dof_kinds`` describe its
            element DOFs.
        mapping: Mapping providing ``jacobian_at(xi, element_ids)``.

    Returns:
        DofTransformation: :class:`IdentityDofTransformation` when every DOF
        is a nodal value, :class:`NodalDerivativeDofTransformation` otherwise.
    """
    if sf.dof_kinds is None or all(kind == "value" for kind in sf.dof_kinds):
        return IdentityDofTransformation()

    return NodalDerivativeDofTransformation(mapping, sf.dof_kinds, sf.dof_nodes_xi)
