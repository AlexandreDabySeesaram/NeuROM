from contextlib import contextmanager
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
            **Must impose zero.** The same constraint object is handed to every
            mode (see :class:`CPPGD`), so a non-zero imposed value ``u_0`` gives
            ``u(x_0) = u_0 * sum_m prod_{k>=1} w_m^k``, which equals ``u_0``
            only if that parametric sum happens to be 1 -- nothing enforces it,
            and no training can fix it since the imposed DOFs are not trainable.
            Rank 1 with unit parametric factors is the only case that works by
            accident. Impose a non-zero BC by *lifting* instead (see the
            ``Non-zero Dirichlet`` note on :class:`CPPGD`).
        init_values (torch.Tensor): Initial nodal values for mode 0 (and every
            mode, if ``init_values_rest`` is left ``None``), shape (n_nodes, dim).
        init_values_rest (torch.Tensor, optional): Initial nodal values for
            every mode after the first (``m >= 1``), same shape as
            ``init_values``. Defaults to ``None``, which reuses ``init_values``
            for all modes -- the previous, uniform behaviour.
    """

    name: str
    nodes_positions: Field
    sf: ShapeFunction
    mapping: object
    quad: QuadratureRule
    constraint: Constraint
    init_values: torch.Tensor
    init_values_rest: torch.Tensor = None

    @property
    def topology(self) -> Topology:
        return self.nodes_positions.topology

    def __post_init__(self):
        # Build the interpolation geometry once, here, so it is a first-class
        # attribute other fields (e.g. a load) can share via `axis.context`.
        # INVARIANT: the Axis builds the *context*, never the *mapping* — the
        # mapping stays injected so a future sub/super-parametric element can use
        # a geometry shape function distinct from the field's `sf`.
        self.mesh = Mesh(self.topology, self.nodes_positions)
        self.context = QuadratureContext(self.mesh, self.quad, self.mapping)


class CPPGD(TensorDecomposition):
    """Canonical-polyadic PGD separated-representation model.

    Represents ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` over ``l`` axes. Holds the
    monoms ``w_m^k`` as ``TrainableField`` on each axis, manages greedy mode
    enrichment, fills a ``FieldLayout`` from its own flagged
    ``QuadratureAssembly`` grid, and exposes the active monom field names
    (:meth:`directory`), matched-pointwise
    inference (:meth:`evaluate`) and a full-tensor grid (:meth:`assemble`). It
    computes no energy and owns no training loop.

    Non-zero Dirichlet: NOT supported, use a lift
        Every mode shares its axis' ``constraint`` object, so a non-zero imposed
        value is written into *every* mode's monom and the reconstruction at that
        node is ``u_0 * sum_m prod_{k>=1} w_m^k``, not ``u_0``. The boundary
        value is therefore wrong at rank > 1 and drifts as the greedy loop
        enriches. It is silent: the constrained DOFs are excluded from
        ``values_reduced``, so nothing in training or in the diagnostics ever
        looks at them.

        The fix is the standard one -- split ``u = u_lift + u_tilde`` with
        ``u_lift`` a fixed field carrying the boundary data and ``u_tilde``
        decomposed under strictly homogeneous constraints. Not implemented:
        ``Axis`` exposes one constraint for all modes, so "mode 0 inhomogeneous,
        modes >= 1 homogeneous" is not expressible yet.

        This is also the root of the restriction on
        :meth:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD.renormalise`.

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
            raise ValueError("CP-PGD admits at most one vector-valued factor per mode.")
        self.n_modes_max = n_modes_max

        # Contexts are built and owned by the axes (Axis.__post_init__). Keep a
        # reference ModuleList only so nn.Module registers them (.to(device),
        # state_dict); dedup by identity happens in IntegrationDomain.
        self._contexts = nn.ModuleList([a.context for a in self.axes])

        # Grid of monoms: modes x axes of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{self.name}_dim{a.name}_mode{m}",
                            topology=a.topology,
                            init_values=(
                                a.init_values
                                if m == 0 or a.init_values_rest is None
                                else a.init_values_rest
                            ),
                            constraint=a.constraint,
                        )
                        for a in self.axes
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # One QuadratureAssembly per monom, grouped into mode-blocks. The leading
        # `n_ini` blocks start active; the rest inactive. `active` is the single
        # source of truth for truncation (n_modes_truncated counts leading active
        # blocks) — no separate counter to keep in sync.
        n_ini = min(n_modes_ini, n_modes_max)
        self._assemblies = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QuadratureAssembly(
                            a.context, a.sf, self.monoms[m][k], active=(m < n_ini)
                        )
                        for k, a in enumerate(self.axes)
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(self.n_modes_truncated):
            self.unfreeze_mode(m)

    @property
    def n_modes_truncated(self) -> int:
        """Number of active modes: the leading run of active mode-blocks.

        Single source of truth is the assemblies' `active` flags. Active blocks
        are contiguous from index 0 because the greedy lifecycle is monotone —
        a mode, once activated, is never deactivated.
        """
        n = 0
        for block in self._assemblies:
            if not bool(block[0].active):
                break
            n += 1
        return n

    def assemblies(self):
        """Flat list of this decomposition's QuadratureAssembly, one per monom.

        Mode-major then axis order. The seam a caller uses to build the shared
        ``IntegrationDomain([*pgd.assemblies(), other_assembly])``.
        """
        return [a for block in self._assemblies for a in block]

    def freeze_all(self):
        """Freeze the monoms of every mode."""
        for m in range(self.n_modes_max):
            self.freeze_mode(m)

    def freeze_mode(self, m):
        """Freeze the monoms of mode ``m``."""
        for k in range(len(self.axes)):
            self.freeze_monom(m, k)

    def unfreeze_mode(self, m):
        """Unfreeze the monoms of mode ``m``."""
        for k in range(len(self.axes)):
            self.unfreeze_monom(m, k)

    def freeze_monom(self, m, k):
        """Freeze mode ``m``'s monom on axis ``k`` alone.

        The per-axis seam, of which :meth:`freeze_mode` is the loop over every
        axis. A strategy that holds one direction fixed while fitting the others
        -- alternating directions, or a schedule that pins the space factor as a
        fixed *support* and retrains only the parametric ones -- goes through
        here rather than reaching into ``monoms[m][k].values_reduced`` itself.

        Args:
            m (int): Mode index.
            k (int): Axis index, in ``self.axes`` order.
        """
        self.monoms[m][k].values_reduced.requires_grad_(False)

    def unfreeze_monom(self, m, k):
        """Unfreeze mode ``m``'s monom on axis ``k`` alone.

        See :meth:`freeze_monom`.
        """
        self.monoms[m][k].values_reduced.requires_grad_(True)

    def add_mode(self):
        """Enrich the decomposition with one new mode (greedy PGD).

        Activates the next mode-block's assemblies, then unfreezes its monoms
        (activate-before-unfreeze, so the mode never passes through the illegal
        active=False/requires_grad=True state). Leaves the freeze state of the
        currently-active modes untouched. Returns the new mode index. Raises
        RuntimeError at capacity.

        The new mode keeps its ``Axis.init_values`` seed rather than being zeroed:
        an all-zero mode is a stationary point of the energy (every gradient
        component is proportional to the *other* factor, so both stay locked at
        0), which never takes off under a gradient optimizer. A non-zero
        parametric seed lets the linear load term drive the enrichment.
        """
        m = self.n_modes_truncated
        if m >= self.n_modes_max:
            raise RuntimeError("Cannot add a mode: all modes are already active.")
        for assembly in self._assemblies[m]:
            assembly.activate()
        self.unfreeze_mode(m)
        return m

    @contextmanager
    def truncated(self, n_modes):
        """Temporarily evaluate this decomposition at rank ``n_modes``.

        Greedy enrichment is nested: a rank-``N`` decomposition contains its own
        rank-1..``N`` approximations, since mode ``m`` was trained against the
        residual of the ``m`` before it. This is the seam that lets a *finished*
        run be scored at every intermediate rank without retraining -- the whole
        error-vs-rank curve from one training run.

        Deactivates the trailing mode-blocks on entry and restores exactly the
        previous active set on exit, including when the body raises. Nothing
        else is touched: the monoms, the coefficient rows and their
        ``requires_grad`` flags are left alone, so this changes what is
        *evaluated*, never what was trained.

        Not for use during training -- the mode lifecycle is monotone, and
        ``add_mode`` appends at ``n_modes_truncated``, so a stage entered inside
        this block would overwrite a trained mode. Read-only inspection only::

            with problem.pgd.truncated(3):
                errors = relative_errors(problem.pgd)

        Args:
            n_modes (int): rank to evaluate at, in ``1..n_modes_truncated``.
                Asking for the current rank is a no-op.

        Yields:
            CPPGD: self, at the requested rank.

        Raises:
            ValueError: if ``n_modes`` is outside ``1..n_modes_truncated``.
                Truncation cannot *add* modes -- a rank above what was trained
                does not exist, and silently returning the trained rank would
                put a mislabelled point on the curve.
        """
        n_active = self.n_modes_truncated
        if not 1 <= n_modes <= n_active:
            raise ValueError(
                f"cannot evaluate at rank {n_modes}: this decomposition has "
                f"{n_active} active mode(s), and truncation only removes them."
            )
        for m in range(n_modes, n_active):
            for assembly in self._assemblies[m]:
                assembly.deactivate()
        try:
            yield self
        finally:
            for m in range(n_modes, n_active):
                for assembly in self._assemblies[m]:
                    assembly.activate()

    def renormalise(self):
        """Fix the representation's scale gauge in place. No-op for CP-PGD.

        The seam a trainer calls at a stage boundary. ``CPPGD`` has the same
        ``d - 1``-dimensional scale degeneracy per mode as its subclasses (the
        product ``prod_k w_m^k`` is unchanged by any per-axis rescaling with
        ``prod_k s_k = 1``), but correcting it here would change every existing
        CP-PGD result, so this deliberately does nothing.
        :class:`~neurom.decompositions.polynomial_pgd.PolynomialNLPGD` overrides
        it.
        """

    def mode_parameters(self, m=None):
        """Return mode ``m``'s trainable monom parameters (optimizer-agnostic).

        The decomposition owns which tensors make up a mode; wiring them into an
        optimizer is :meth:`neurom.neurom_model.NeuROMModel.add_mode_to_optimizer`'s
        job, so the PGD stays agnostic to the optimizer.

        Args:
            m (int, optional): Index of the mode. Supports negative indexing
                (Python-style). Defaults to the last-activated mode
                (``n_modes_truncated - 1``).

        Returns:
            list[torch.Tensor]: The monom parameters of mode ``m``, one per axis.

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
        return [f.values_reduced for f in self.monoms[m]]

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
                + (
                    f" of shape {tuple(points.shape)}"
                    if torch.is_tensor(points)
                    else ""
                )
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
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1))  # (P, 1, dim_k)
                w = w.reshape(w.shape[0], -1)  # (P, dim_k)
                prod = w if prod is None else prod * w  # scalar * vector broadcasts
            total = prod if total is None else total + prod
        return total

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
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1)).reshape(
                    P_k, -1
                )  # (N_k, d_k)
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
