"""Build the FE interpolant of a known function: its DOF values on a space.

This is the classical finite-element interpolation operation -- given a function ``f`` and a
:class:`~neurom.function_space.FunctionSpace`, produce the DOF values whose FE field represents
``f`` (``value`` DOFs get :math:`f(x)`, ``d1`` DOFs get :math:`f'(x)` via autograd).

It is an **optional** utility, separate from field construction: a :class:`TrainableField`'s
initial values are a seed for the optimiser and need *not* be any function's interpolant, so a
field is always built from raw values. Only point-functional elements (Lagrange, Hermite) are
supported; a DOF kind without a functional (``flux``/``normal_derivative``) raises.
"""

import torch

from neurom.dof_transformations.functionals import FUNCTIONALS
from neurom.meshes.topology import EntityDim


def _dof_kinds_by_dim(element):
    """Map ``entity_dim -> ordered list of DOF kinds`` on one entity of that dimension."""
    out: dict[int, list] = {}
    for entity_dim, _, kind in element.dof_layout:
        out.setdefault(entity_dim, [])
        if kind not in out[entity_dim]:
            out[entity_dim].append(kind)
    return out


def _entity_points(topology, coordinates, entity_dim):
    """Physical coordinates of every entity of ``entity_dim`` (where its DOFs sit)."""
    pos = coordinates.full_values()  # (n_vertices, gdim)
    if entity_dim == EntityDim.VERTEX:
        return pos
    if entity_dim == EntityDim.EDGE and topology.dim >= 2:  # edge midpoint
        return pos[topology.edge_vertices].mean(dim=1)
    if entity_dim == topology.cell_dim:  # cell centroid
        return pos[topology.cell_vertices].mean(dim=1)
    raise NotImplementedError("face DOF points not built yet")


def interpolate(space, mesh, f) -> torch.Tensor:
    """Build the FE interpolant of ``f`` on ``space``: its DOF values.

    Each DOF kind applies its functional to ``f`` -- ``value`` :math:`\\to f(x)`, ``d1``
    :math:`\\to f'(x)` (autograd) -- so a single callable yields the DOF values whose FE field
    represents ``f``.

    Args:
        space (FunctionSpace): The target space.
        mesh (Mesh): The mesh placing the DOFs (must share ``space.topology``).
        f (callable): ``x (n, gdim) -> value (n[, *value_shape])``.

    Returns:
        torch.Tensor: DOF values of shape ``(n_scalar_dofs, n_components)`` (detached).

    Raises:
        ValueError: If ``mesh`` is defined over a different topology than ``space``.
        NotImplementedError: If the element has a DOF kind with no functional (e.g. a
            ``flux``/``normal_derivative`` element) -- the FE interpolant of a function is only
            defined for point-functional elements so far.
    """
    if mesh.topology is not space.topology:
        raise ValueError("interpolate: mesh and space must share a topology.")
    n_comp = space.element.n_components
    coordinates = mesh.coordinates
    out = torch.zeros(
        space.n_scalar_dofs, n_comp, dtype=coordinates.full_values().dtype
    )
    for entity_dim, kinds in _dof_kinds_by_dim(space.element).items():
        x = _entity_points(
            space.topology, coordinates, entity_dim
        )  # (n_entities, gdim)
        entities = range(x.shape[0])
        for kind in kinds:
            if kind not in FUNCTIONALS:
                raise NotImplementedError(
                    f"interpolate: no functional for DOF kind {kind!r}; the FE interpolant "
                    "of a function is only defined for point-functional elements "
                    "(Lagrange, Hermite) so far."
                )
            vals = FUNCTIONALS[kind].on_function(f, x).reshape(x.shape[0], n_comp)
            out[space.dof_map.locate(entity_dim, entities, [kind])] = vals.detach()
    return out
