r"""Build the FE interpolant of a known function: its DOF values on a space.

This is the classical finite-element interpolation operation -- given a function ``f`` and a
:class:`~neurom.function_space.FunctionSpace`, produce the DOF values whose FE field represents
``f`` (``value`` DOFs get :math:`f(x)`, ``d1`` DOFs get :math:`f'(x)` via autograd).

Each DOF is evaluated at its *own* reference node (``element.dof_nodes_xi``), mapped to physical
space per cell, and its functional applied there; the result is scattered to the DOF's global index
through the DOF map's ``cell_dofs``. Because a shared DOF lands on the same physical point from
every incident cell -- and the orientation permutation sends each cell's local DOF to the
matching global index -- the scatter is single-valued, so an entity carrying several DOFs (e.g.
:math:`P_3`'s two edge nodes) is handled with no special casing.

A **point** functional (``value``/``d1``) is driven through ``on_function`` at its physical node; an
**integral** functional (``flux``, ``acts_pointwise=False``) instead gets the facet normal/measure
gathered from the mesh and is driven through its entity-aware path (e.g. RT0:
:math:`|E|\,\mathbf f(\mathbf m)\cdot\hat{\mathbf n}`). A DOF kind with *no* functional
(``normal_derivative``) still raises.

It is an **optional** utility, separate from field construction: a :class:`TrainableField`'s
initial values are a seed for the optimiser and need *not* be any function's interpolant, so a
field is always built from raw values.
"""

import torch

from neurom.samplings import QuadratureSampling


def _functionals_by_dof(element):
    """Group local DOF indices by their functional: ``functional -> [local index, ...]``.

    DOFs sharing a functional (all ``value`` DOFs, say) are evaluated in one batched call; the
    grouping is by functional *identity* (``kind`` maps to a shared instance), so same-kind DOFs
    on different entities still batch together.
    """
    by_func: dict = {}
    for i, dof in enumerate(element.dof_layout):
        by_func.setdefault(dof.functional, []).append(i)
    return by_func


def interpolate(space, mesh, f) -> torch.Tensor:
    """Build the FE interpolant of ``f`` on ``space``: its DOF values.

    Each DOF's functional is applied to ``f`` at the DOF's physical node -- ``value``
    :math:`\\to f(x)`, ``d1`` :math:`\\to f'(x)` (autograd) -- so a single callable yields the DOF
    values whose FE field represents ``f``.

    Args:
        space (FunctionSpace): The target space.
        mesh (Mesh): The mesh placing the DOFs (must share ``space.topology``); its P1 coordinate
            field supplies the reference-to-physical map.
        f (callable): ``x (n, gdim) -> value (n, n_components)`` -- the component axis is explicit
            (a scalar field is ``(n, 1)``). A scalar ``f`` returning ``(n,)`` is also accepted and
            normalised to the explicit axis.

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
    element = space.element
    if any(d.functional is None for d in element.dof_layout):
        raise NotImplementedError(
            "interpolate: a DOF kind has no functional (e.g. normal_derivative); the FE interpolant "
            "of a function is only defined for elements whose DOF kinds have a functional."
        )

    from neurom.geometry import AffineSimplexMapping, EntityGeometry

    n_comp = element.n_components
    dtype = mesh.coordinates.full_values().dtype
    n_cells = mesh.topology.n_cells

    # Physical node of every DOF, per cell: map each DOF's reference coordinate through the cell's
    # (affine) geometry. The DOF axis rides the quadrature slot of the sampling (batch-agnostic map).
    xi = element.dof_nodes_xi().to(dtype)  # (n_dofs, dim_ref)
    pts = QuadratureSampling(
        xi.unsqueeze(0).expand(n_cells, *xi.shape)
    )  # (n_cells, n_dofs, dim_ref)
    x_phys = AffineSimplexMapping(mesh).map(pts).values  # (n_cells, n_dofs, gdim)
    gdim = x_phys.shape[-1]

    cell_dofs = space.dof_map.cell_dofs  # (n_cells, n_dofs) global indices
    out = torch.zeros(space.n_scalar_dofs, n_comp, dtype=dtype)
    entity_geom = (
        None  # built lazily, only if an integral (non-point) functional needs it
    )
    for functional, cols in _functionals_by_dof(element).items():
        if functional.acts_pointwise:
            pts_f = x_phys[:, cols, :].reshape(-1, gdim)  # (n_cells * len(cols), gdim)
            vals = functional.on_function(f, pts_f).reshape(n_cells, len(cols), n_comp)
        else:
            # Integral (flux/moment) functional: it needs the facet normal and measure, gathered
            # here where the mesh is known (not a single point). The canonical (global) facet
            # geometry makes the scatter single-valued across a shared facet; T_e handles the sign.
            entity_geom = entity_geom or EntityGeometry(mesh)
            facets = [
                element.dof_layout[c].local_index for c in cols
            ]  # local facet of each DOF
            normals = entity_geom.facet_normals()[
                :, facets, :
            ]  # (n_cells, len(cols), gdim)
            measures = entity_geom.facet_measures()[
                :, facets, :
            ]  # (n_cells, len(cols), 1)
            vals = functional.flux_values(f, x_phys[:, cols, :], normals, measures)
        # Scatter to global DOFs; shared DOFs receive the same value from every incident cell.
        out[cell_dofs[:, cols]] = vals.detach()
    return out
