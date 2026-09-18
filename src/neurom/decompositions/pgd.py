import string

import torch
import torch.nn as nn

from neurom.fields.trainable_field import TrainableField
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator
from neurom.decompositions.tensor_decomposition import TensorDecomposition


class CPPGD(TensorDecomposition):
    """Canonical-polyadic PGD separated-representation model.

    Represents ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` over ``l`` factors. Holds
    the monoms ``w_m^k`` as ``TrainableField`` on each factor, manages greedy mode
    enrichment, fills a ``FieldLayout`` from its own flagged
    ``QuadratureAssembly`` grid, and exposes the active monom field names
    (:meth:`directory`), matched-pointwise
    inference (:meth:`evaluate`) and a full-tensor grid (:meth:`assemble`). It
    computes no energy and owns no training loop.

    What it is built from, and what it builds
        The caller supplies one
        :class:`~neurom.decompositions.factor.MonomSpec` per factor -- ``l`` of
        them, whatever the number of modes -- each posed on a
        :class:`~neurom.decompositions.factor.FactorSpace`. From that,
        ``__init__`` builds the ``n_modes_max x l`` grid of monoms: one
        ``TrainableField`` per (mode, spec) pair, seeded from the spec's
        ``init_values`` and sharing its ``constraint``, plus one
        ``QuadratureAssembly`` each. Every monom of factor ``k`` therefore binds
        to the same ``FactorSpace`` and, through it, to a single
        ``QuadratureContext`` -- which is why ``assemblies()`` returns
        ``n_modes_max * l`` assemblies over only ``l`` distinct contexts, and why
        ``IntegrationDomain`` updates that geometry once per forward instead of
        once per monom.

        Enriching (:meth:`add_mode`) adds a row to that grid. It never creates a
        ``MonomSpec`` or a ``FactorSpace``; it reads the existing ones again.

    Evaluation: diagonal (:meth:`evaluate`) vs grid (:meth:`assemble`)
        Two ways to sample the trained field, with **different input formats and
        different semantics** -- pick by whether you want paired points or every
        crossing:

        * :meth:`evaluate` -- *diagonal*. Input: a single ``(P, n_factors)`` tensor,
          one **point per row** (``pts[p] == (x_p, E_p, ...)``). Evaluates the
          ``P`` given tuples and returns ``(P, d)``. Use it for a cloud of
          arbitrary query points, or a 1-D slice (e.g. fix E, sweep x by pairing
          each x with the same E). All factor columns therefore share the length P.
        * :meth:`assemble` -- *grid*. Input: a **list** of one 1-D tensor per
          factor, lengths **independent** ``(N_1, ..., N_l)``. Evaluates **every**
          combination (tensor product) and returns ``(N_1, ..., N_l[, d])``. Use
          it for a full parametric surface / heatmap over ``x`` x ``E``.

        The trailing ``d`` is the single vector factor's dim (dropped if all
        factors are scalar). Both sum over modes and detach.

    Args:
        monom_specs (list[MonomSpec]): One blueprint per factor, in order. Each
            one is read ``n_modes_max`` times, to build that factor's monom for
            every mode; the decomposition creates the ``TrainableField``, never
            the specs.
        n_modes_max (int): Maximum number of modes.
        name (str): Prefix for the monom field names
            (``f"{name}_dim{spec.space.name}_mode{m}"``); makes them unique so several
            decompositions can share one ``FieldLayout``.
        n_modes_ini (int): Number of initially active (trainable) modes.
    """

    def __init__(self, monom_specs, n_modes_max, name="pgd", n_modes_ini=1):
        super().__init__()
        self.name = name
        self.monom_specs = list(monom_specs)
        if sum(int(s.init_values.shape[1] > 1) for s in self.monom_specs) > 1:
            raise ValueError("CP-PGD admits at most one vector-valued factor per mode.")
        self.n_modes_max = n_modes_max

        # Contexts are built and owned by the factor spaces
        # (FactorSpace.__post_init__). Keep a reference ModuleList only so
        # nn.Module registers them (.to(device), state_dict); dedup by identity
        # happens in IntegrationDomain. Several factors may share one space, so
        # this list can hold duplicates.
        self._contexts = nn.ModuleList([s.space.context for s in self.monom_specs])

        # Grid of monoms: modes x factors of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{self.name}_dim{spec.space.name}_mode{m}",
                            connectivity=spec.space.connectivity,
                            init_values=spec.init_values,
                            constraint=spec.constraint,
                        )
                        for spec in self.monom_specs
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # One QuadratureAssembly per monom, grouped into mode-blocks. The leading
        # `n_ini` blocks start active; the rest inactive. `active` is the single
        # source of truth for truncation (n_active_modes counts leading active
        # blocks) — no separate counter to keep in sync.
        n_ini = min(n_modes_ini, n_modes_max)
        self._assemblies = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QuadratureAssembly(
                            spec.space.context,
                            spec.sf,
                            self.monoms[m][k],
                            active=(m < n_ini),
                        )
                        for k, spec in enumerate(self.monom_specs)
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(self.n_active_modes):
            self.unfreeze_mode(m)

    @property
    def n_active_modes(self) -> int:
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

        Mode-major then factor order. The seam a caller uses to build the shared
        ``IntegrationDomain([*pgd.assemblies(), other_assembly])``.
        """
        return [a for block in self._assemblies for a in block]

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

        Activates the next mode-block's assemblies, then unfreezes its monoms
        (activate-before-unfreeze, so the mode never passes through the illegal
        active=False/requires_grad=True state). Leaves the freeze state of the
        currently-active modes untouched. Returns the new mode index. Raises
        RuntimeError at capacity.

        The new mode keeps its ``MonomSpec.init_values`` seed rather than being zeroed:
        an all-zero mode is a stationary point of the energy (every gradient
        component is proportional to the *other* factor, so both stay locked at
        0), which never takes off under a gradient optimizer. A non-zero
        parametric seed lets the linear load term drive the enrichment.
        """
        m = self.n_active_modes
        if m >= self.n_modes_max:
            raise RuntimeError("Cannot add a mode: all modes are already active.")
        for assembly in self._assemblies[m]:
            assembly.activate()
        self.unfreeze_mode(m)
        return m

    def mode_parameters(self, m=None):
        """Return mode ``m``'s trainable monom parameters (optimizer-agnostic).

        The decomposition owns which tensors make up a mode; wiring them into an
        optimizer is :meth:`neurom.neurom_model.NeuROMModel.add_mode_to_optimizer`'s
        job, so the PGD stays agnostic to the optimizer.

        Args:
            m (int, optional): Index of the mode. Supports negative indexing
                (Python-style). Defaults to the last-activated mode
                (``n_active_modes - 1``).

        Returns:
            list[torch.Tensor]: The monom parameters of mode ``m``, one per factor.

        Raises:
            IndexError: If ``m`` is out of range for the active modes.
        """
        n_active = int(self.n_active_modes)
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
        """Register every monom field (all modes, all factors) in the layout.

        Called once at setup. All ``n_modes_max`` modes are registered up front
        (including not-yet-active ones) so ``add_mode`` needs no layout
        reference. Inactive monoms are registered but never interpolated.
        """
        for mode in self.monoms:
            for field in mode:
                field_layout.add(field)

    def directory(self):
        """Ordered lookup table of active monom field names, keyed by factor.

        Returns:
            dict[str, list[str]]: factor name -> monom field names, one per active
            mode (index ``m``). Feed to a physics/energy term to read each monom
            out of the FieldLayout by name (``field_layout[name]``). Truncates to
            active modes; grows after :meth:`add_mode`.
        """
        n = self.n_active_modes
        return {
            spec.space.name: [self.monoms[m][k].name for m in range(n)]
            for k, spec in enumerate(self.monom_specs)
        }

    def _as_factor_columns(self, points):
        """Split the ``(P, n_factors)`` query-point tensor into per-factor columns.

        The public form is a single ``(P, n_factors)`` tensor, one **point per
        row** (``points[p] == (x_p, E_p, ...)``) -- the natural, human-friendly
        form. Internally the evaluation consumes one column per factor, so this
        just unbinds the columns (the transpose). Returns a list of
        ``n_factors`` 1-D tensors of length ``P``.
        """
        if not torch.is_tensor(points) or points.dim() != 2:
            raise ValueError(
                "evaluate expects a (P, n_factors) tensor of query points, one "
                "point "
                f"per row; got {type(points).__name__}"
                + (
                    f" of shape {tuple(points.shape)}"
                    if torch.is_tensor(points)
                    else ""
                )
                + "."
            )
        if points.shape[1] != len(self.monom_specs):
            raise ValueError(
                f"Query points have {points.shape[1]} columns but the "
                f"decomposition has {len(self.monom_specs)} factors; each row must "
                "be a full coordinate tuple (one value per factor)."
            )
        return list(points.T)

    def evaluate(self, coords):
        """Evaluate ``u`` at matched query points (diagonal), summed over modes.

        Args:
            coords (torch.Tensor): ``(P, n_factors)`` tensor of the ``P`` query
                points, one **point per row** (``coords[p] == (x_p, E_p, ...)``).
                This is "matched/diagonal": each row is one full coordinate tuple,
                evaluated as a single point; there is no grid (see
                :meth:`assemble` for the tensor-product grid).

        Returns:
            torch.Tensor: ``(P, d)`` = ``sum_m prod_k w_m^k(coords[p, k])``; ``d``
            is the single vector factor's dim, or 1 if all factors are scalar.
            Detached (via ``PointWiseInterpolator``).
        """
        coords = self._as_factor_columns(coords)
        n = self.n_active_modes
        total = None
        for m in range(n):
            prod = None
            for k, spec in enumerate(self.monom_specs):
                pwi = PointWiseInterpolator(
                    spec.space.mesh,
                    spec.sf,
                    self.monoms[m][k],
                    spec.space.mapping,
                )
                w = pwi.at_position(coords[k].reshape(-1, 1))  # (P, dim_k)
                prod = w if prod is None else prod * w  # scalar * vector broadcasts
            total = prod if total is None else total + prod
        return total

    def assemble(self, coords):
        """Assemble the full separated tensor at the given per-factor coordinates.

        Args:
            coords (list[torch.Tensor]): One 1-D tensor per factor (length
                N_k), the query coordinates on that factor.

        Returns:
            torch.Tensor: full grid tensor of shape ``(N_1, ..., N_l[, d])``; the
            trailing ``d`` is present iff a vector factor exists (else dropped).
            Equals ``sum_m prod_k w_m^k`` over the coordinate grid. Detached.
        """
        n_modes = self.n_active_modes
        mode_letter = "Z"
        per_factor = []  # per_factor[k]: (n_modes, N_k) or (n_modes, N_k, d_k)
        for k, spec in enumerate(self.monom_specs):
            cols = []
            for m in range(n_modes):
                pwi = PointWiseInterpolator(
                    spec.space.mesh,
                    spec.sf,
                    self.monoms[m][k],
                    spec.space.mapping,
                )
                w = pwi.at_position(coords[k].reshape(-1, 1))  # (N_k, d_k)
                cols.append(w.reshape(-1) if w.shape[1] == 1 else w)
            per_factor.append(torch.stack(cols, dim=0))

        grid_letters = string.ascii_lowercase[: len(self.monom_specs)]
        comp_pool = iter(c for c in string.ascii_uppercase if c != mode_letter)
        in_subs, out_grid, out_comp = [], "", ""
        for k, arr in enumerate(per_factor):
            sub = mode_letter + grid_letters[k]
            out_grid += grid_letters[k]
            if arr.dim() == 3:  # vector factor: (n_modes, N_k, d_k)
                c = next(comp_pool)
                sub += c
                out_comp += c
            in_subs.append(sub)
        return torch.einsum(f"{','.join(in_subs)}->{out_grid}{out_comp}", *per_factor)
