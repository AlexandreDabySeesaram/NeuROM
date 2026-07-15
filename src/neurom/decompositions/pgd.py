from dataclasses import dataclass
import string

import torch
import torch.nn as nn

from neurom.fields.field import Field
from neurom.fields.trainable_field import TrainableField
from neurom.shape_functions.shape_function import ShapeFunction
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.constraints.constraint import Constraint
from neurom.meshes.topology import Topology
from neurom.meshes.mesh import Mesh
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.separated_domain import SeparatedDomain
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.decompositions.base import TensorDecomposition


@dataclass
class Axis:
    """Descriptor of one factor (coordinate direction) of a CP-PGD decomposition.

    Groups everything needed to build and interpolate the monoms living on this
    axis. The axis mesh topology is derived from ``nodes_positions`` so that the
    ``Mesh`` identity check and the monoms' ``TrainableField`` share the exact
    same ``Topology`` object.

    Attributes:
        name (str): Axis name, used as key in the separated interpolation output.
        nodes_positions (Field): Coordinates of the axis mesh nodes.
        sf (ShapeFunction): Shape function used for interpolation on this axis.
        mapping: Reference/physical mapping (e.g. IsoparametricMapping1D).
        quad (QuadratureRule): Quadrature rule for integration on this axis.
        constraint (Constraint): Constraint (boundary conditions) on this axis.
        init_values (torch.Tensor): Initial nodal values for each new monom,
            shape (n_nodes, dim).
    """

    name: str
    nodes_positions: Field
    sf: ShapeFunction
    mapping: object
    quad: QuadratureRule
    constraint: Constraint
    init_values: torch.Tensor

    @property
    def topology(self) -> Topology:
        return self.nodes_positions.topology


class CPPGD(TensorDecomposition):
    """Canonical-polyadic PGD separated-representation model.

    Represents ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` over ``l`` axes. Holds the
    monoms ``w_m^k`` as ``TrainableField`` on each axis, manages greedy mode
    enrichment, fills a ``FieldLayout`` through an owned ``SeparatedDomain``, and
    exposes the active monom field names (:meth:`directory`), matched-pointwise
    inference (:meth:`evaluate`) and a full-tensor grid (:meth:`assemble`). It
    computes no energy and owns no training loop.

    Evaluation: diagonal (:meth:`evaluate`) vs grid (:meth:`assemble`)
        Two ways to sample the trained field, with **different input formats and
        different semantics** -- pick by whether you want paired points or every
        crossing:

        * :meth:`evaluate` -- *diagonal*. Input: a single ``(P, n_axes)`` tensor,
          one **point per row** (``pts[p] == (x_p, E_p, ...)``). Evaluates the
          ``P`` given tuples and returns ``(P, d)``. Use it for a cloud of
          arbitrary query points, or a 1-D slice (e.g. fix E, sweep x by pairing
          each x with the same E). All axis columns therefore share the length P.
        * :meth:`assemble` -- *grid*. Input: a **list** of one 1-D tensor per
          axis, lengths **independent** ``(N_1, ..., N_l)``. Evaluates **every**
          combination (tensor product) and returns ``(N_1, ..., N_l[, d])``. Use
          it for a full parametric surface / heatmap over ``x`` x ``E``.

        The trailing ``d`` is the single vector factor's dim (dropped if all
        factors are scalar). Both sum over modes and detach.

    Args:
        axes (list[Axis]): The ordered axes of the decomposition.
        n_modes_max (int): Maximum number of modes.
        name (str): Prefix for the monom field names
            (``f"{name}_dim{axis.name}_mode{m}"``); makes them unique so several
            decompositions can share one ``FieldLayout``.
        n_modes_ini (int): Number of initially active (trainable) modes.
    """

    def __init__(self, axes, n_modes_max, name="pgd", n_modes_ini=1):
        super().__init__()
        self.name = name
        self.axes = list(axes)
        if sum(int(a.init_values.shape[1] > 1) for a in self.axes) > 1:
            raise ValueError(
                "CP-PGD admits at most one vector-valued factor per mode."
            )
        self.n_modes_max = n_modes_max

        # One Mesh + QuadratureContext per axis, shared across modes.
        self._meshes = nn.ModuleList(
            [Mesh(a.topology, a.nodes_positions) for a in self.axes]
        )
        self._contexts = nn.ModuleList(
            [
                QuadratureContext(mesh, a.quad, a.mapping)
                for mesh, a in zip(self._meshes, self.axes)
            ]
        )

        # Grid of monoms: modes x axes of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{self.name}_dim{a.name}_mode{m}",
                            topology=a.topology,
                            init_values=a.init_values,
                            constraint=a.constraint,
                        )
                        for a in self.axes
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Truncation-aware domain: one assembly per monom, grouped by mode.
        mode_blocks = [
            [
                QuadratureAssembly(self._contexts[k], a.sf, self.monoms[m][k])
                for k, a in enumerate(self.axes)
            ]
            for m in range(self.n_modes_max)
        ]
        self.domain = SeparatedDomain(
            mode_blocks, n_active_modes=min(n_modes_ini, n_modes_max)
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(self.n_modes_truncated):
            self.unfreeze_mode(m)

    @property
    def n_modes_truncated(self) -> int:
        """Number of currently-active modes (single source of truth: the domain)."""
        return int(self.domain.n_active_modes)

    def freeze_all(self):
        """Freeze the monoms of every mode."""
        for m in range(self.n_modes_max):
            self.freeze_mode(m)

    def freeze_mode(self, m):
        """Freeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(False)

    def unfreeze_mode(self, m):
        """Unfreeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(True)

    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Activates the next mode-block in the domain (zeroed out and trainable)
        without touching the freeze state of the currently-active modes. Returns
        the index of the newly-activated mode. Raises RuntimeError at capacity.
        """
        new = self.domain.grow()
        self._zero_out(new)
        self.unfreeze_mode(new)
        return new

    def _zero_out(self, m):
        """Zero the nodal values of every monom of mode ``m``."""
        with torch.no_grad():
            for field in self.monoms[m]:
                field.values_reduced.zero_()

    def add_mode_to_optimizer(self, optim, m=None):
        """Add mode ``m``'s monom parameters to ``optim`` as a new param group.

        Args:
            optim (torch.optim.Optimizer): Optimizer to enrich.
            m (int, optional): Index of the mode to add. Supports negative
                indexing (Python-style). Defaults to the last-activated mode
                (``n_modes_truncated - 1``).

        Raises:
            IndexError: If ``m`` is out of range for the active modes.
        """
        n_active = int(self.n_modes_truncated)
        if m is None:
            m = n_active - 1
        if m < 0:
            m += n_active
        if not 0 <= m < n_active:
            raise IndexError(
                f"Mode index {m} out of range for {n_active} active mode(s)."
            )
        params = [f.values_reduced for f in self.monoms[m]]
        optim.add_param_group({"params": params})

    def register_into(self, field_layout):
        """Register every monom field (all modes, all axes) in the layout.

        Called once at setup. All ``n_modes_max`` modes are registered up front
        (including not-yet-active ones) so ``add_mode`` needs no layout
        reference. Inactive monoms are registered but never interpolated.
        """
        for mode in self.monoms:
            for field in mode:
                field_layout.add(field)

    def directory(self):
        """Ordered lookup table of active monom field names, keyed by axis.

        Returns:
            dict[str, list[str]]: axis name -> monom field names, one per active
            mode (index ``m``). Feed to a physics/energy term to read each monom
            out of the FieldLayout by name (``field_layout[name]``). Truncates to
            active modes; grows after :meth:`add_mode`.
        """
        n = self.n_modes_truncated
        return {
            axis.name: [self.monoms[m][k].name for m in range(n)]
            for k, axis in enumerate(self.axes)
        }

    def _as_axis_columns(self, points):
        """Split the ``(P, n_axes)`` query-point tensor into per-axis columns.

        The public form is a single ``(P, n_axes)`` tensor, one **point per
        row** (``points[p] == (x_p, E_p, ...)``) -- the natural, human-friendly
        form. Internally the evaluation consumes one column per axis, so this
        just unbinds the columns (the transpose). Returns a list of ``n_axes``
        1-D tensors of length ``P``.
        """
        if not torch.is_tensor(points) or points.dim() != 2:
            raise ValueError(
                "evaluate expects a (P, n_axes) tensor of query points, one point "
                f"per row; got {type(points).__name__}"
                + (f" of shape {tuple(points.shape)}" if torch.is_tensor(points) else "")
                + "."
            )
        if points.shape[1] != len(self.axes):
            raise ValueError(
                f"Query points have {points.shape[1]} columns but the "
                f"decomposition has {len(self.axes)} axes; each row must be a "
                "full coordinate tuple (one value per axis)."
            )
        return list(points.T)

    def evaluate(self, coords):
        """Evaluate ``u`` at matched query points (diagonal), summed over modes.

        Args:
            coords (torch.Tensor): ``(P, n_axes)`` tensor of the ``P`` query
                points, one **point per row** (``coords[p] == (x_p, E_p, ...)``).
                This is "matched/diagonal": each row is one full coordinate tuple,
                evaluated as a single point; there is no grid (see
                :meth:`assemble` for the tensor-product grid).

        Returns:
            torch.Tensor: ``(P, d)`` = ``sum_m prod_k w_m^k(coords[p, k])``; ``d``
            is the single vector factor's dim, or 1 if all factors are scalar.
            Detached (via ``PointWiseInterpolator``).
        """
        coords = self._as_axis_columns(coords)
        n = self.n_modes_truncated
        total = None
        for m in range(n):
            prod = None
            for k, axis in enumerate(self.axes):
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1))   # (P, 1, dim_k)
                w = w.reshape(w.shape[0], -1)                # (P, dim_k)
                prod = w if prod is None else prod * w       # scalar * vector broadcasts
            total = prod if total is None else total + prod
        return total

    def fill(self, field_layout):
        """Interpolate every active monom and ``update`` it in the layout.

        Delegates to the truncation-aware :class:`SeparatedDomain`: only active
        modes are interpolated. CP analogue of
        ``IntegrationDomain.interpolate_all``.
        """
        self.domain.interpolate_all(field_layout)

    def assemble(self, coords):
        """Assemble the full separated tensor at the given per-axis coordinates.

        Args:
            coords (list[torch.Tensor]): One 1-D tensor per axis (length N_k),
                the query coordinates on that axis.

        Returns:
            torch.Tensor: full grid tensor of shape ``(N_1, ..., N_l[, d])``; the
            trailing ``d`` is present iff a vector factor exists (else dropped).
            Equals ``sum_m prod_k w_m^k`` over the coordinate grid. Detached.
        """
        n_modes = self.n_modes_truncated
        mode_letter = "Z"
        per_axis = []  # per_axis[k]: (n_modes, N_k) or (n_modes, N_k, d_k)
        for k, axis in enumerate(self.axes):
            P_k = coords[k].reshape(-1).shape[0]
            cols = []
            for m in range(n_modes):
                pwi = PointWiseInterpolator(
                    self._meshes[k], axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1)).reshape(P_k, -1)  # (N_k, d_k)
                cols.append(w.reshape(-1) if w.shape[1] == 1 else w)
            per_axis.append(torch.stack(cols, dim=0))

        grid_letters = string.ascii_lowercase[: len(self.axes)]
        comp_pool = iter(c for c in string.ascii_uppercase if c != mode_letter)
        in_subs, out_grid, out_comp = [], "", ""
        for k, arr in enumerate(per_axis):
            sub = mode_letter + grid_letters[k]
            out_grid += grid_letters[k]
            if arr.dim() == 3:  # vector axis: (n_modes, N_k, d_k)
                c = next(comp_pool)
                sub += c
                out_comp += c
            in_subs.append(sub)
        return torch.einsum(f"{','.join(in_subs)}->{out_grid}{out_comp}", *per_axis)
