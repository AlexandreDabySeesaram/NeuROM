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
    the monoms ``w_m^k`` as ``TrainableField`` on each factor, and exposes the
    monom field names (:meth:`directory`), matched-pointwise inference
    (:meth:`evaluate`) and a full-tensor grid (:meth:`assemble`).

    It is a **container, not a strategy**: it computes no energy, owns no
    training loop, and decides nothing about when to enrich or what to
    optimise. It knows how many modes it has and how to grow one more.

    What it is built from, and what it builds
        The caller supplies one
        :class:`~neurom.decompositions.factor.MonomSpec` per factor -- ``l`` of
        them, whatever the number of modes -- each posed on a
        :class:`~neurom.decompositions.factor.FactorSpace`. From that,
        ``__init__`` builds the first ``n_modes_ini`` rows of an
        ``n_modes x l`` grid of monoms: one ``TrainableField`` per (mode, spec)
        pair, seeded from the spec's ``init_values`` and sharing its
        ``constraint``, plus one ``QuadratureAssembly`` each.

        The grid is **dynamic**: :meth:`add_mode` appends a row, reading the
        same ``MonomSpec`` once more. It never creates a ``MonomSpec`` or a
        ``FactorSpace``, and nothing is pre-allocated -- a mode that is never
        added never exists, so it costs no field in the ``FieldLayout``, no
        entry in ``state_dict`` and no interpolation.

        Every monom of factor ``k`` binds to the same ``FactorSpace`` and,
        through it, to a single ``QuadratureContext`` -- including the monoms
        of modes added later. This is why ``assemblies()`` returns
        ``n_modes * l`` assemblies over only ``l`` distinct contexts, and why
        ``IntegrationDomain`` updates that geometry once per forward instead of
        once per monom.

    Freezing: mechanism, not policy
        The ``freeze_*`` / ``unfreeze_*`` methods flip ``requires_grad`` on the
        monoms of a row (:meth:`freeze_mode`), of a column
        (:meth:`freeze_factor`) or of a single cell (:meth:`freeze_monom`).
        They are plain utilities shared by several training strategies --
        progressive greedy freezes the previous rows, simultaneous training
        freezes nothing, alternating-direction minimisation cycles over the
        columns.

        The decomposition calls none of them on its own: a mode is born
        trainable and stays so until someone decides otherwise. That someone is
        the trainer.

        A frozen monom is **still part of u**. ``requires_grad`` says "not
        optimised", never "absent": every monom that exists is interpolated and
        contributes to the energy, which is exactly what all three strategies
        above need.

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
            one is read once per mode, to build that factor's monom; the
            decomposition creates the ``TrainableField``, never the specs.
        name (str): Prefix for the monom field names
            (``f"{name}_dim{spec.space.name}_mode{m}"``); makes them unique so several
            decompositions can share one ``FieldLayout``.
        n_modes_ini (int): Number of modes to build at construction. There is
            no maximum: how far to enrich is the trainer's business.
    """

    def __init__(self, monom_specs, name="pgd", n_modes_ini=1):
        super().__init__()
        self.name = name
        self.monom_specs = list(monom_specs)
        if sum(int(s.init_values.shape[1] > 1) for s in self.monom_specs) > 1:
            raise ValueError("CP-PGD admits at most one vector-valued factor per mode.")

        # Contexts are built and owned by the factor spaces
        # (FactorSpace.__post_init__). Keep a reference ModuleList only so
        # nn.Module registers them (.to(device), state_dict); dedup by identity
        # happens in IntegrationDomain. Several factors may share one space, so
        # this list can hold duplicates.
        self._contexts = nn.ModuleList([s.space.context for s in self.monom_specs])

        # Grid of monoms: modes x factors, grown one row at a time. Empty here;
        # add_mode() is the single code path that builds a mode, so the initial
        # modes and the enriched ones cannot drift apart.
        self.monoms = nn.ModuleList()
        self._assemblies = nn.ModuleList()
        for _ in range(n_modes_ini):
            self.add_mode()

    @property
    def n_modes(self) -> int:
        """Number of modes currently in the decomposition."""
        return len(self.monoms)

    def assemblies(self):
        """Flat list of this decomposition's QuadratureAssembly, one per monom.

        Mode-major then factor order; ``n_modes * l`` long, and one row longer
        after every :meth:`add_mode`. Because it grows, it is not stored in the
        ``IntegrationDomain`` -- the caller passes it per forward, as
        ``domain.interpolate_all(layout, decomposition.assemblies())``.
        """
        return [a for block in self._assemblies for a in block]

    def _set_requires_grad(self, fields, flag):
        for field in fields:
            field.values_reduced.requires_grad_(flag)

    def freeze_all(self):
        """Freeze every monom of the grid."""
        self._set_requires_grad((f for row in self.monoms for f in row), False)

    def unfreeze_all(self):
        """Unfreeze every monom of the grid."""
        self._set_requires_grad((f for row in self.monoms for f in row), True)

    def freeze_mode(self, m):
        """Freeze the monoms of mode ``m`` -- a **row** of the grid.

        What progressive greedy PGD does to the already-converged modes before
        enriching. The mode keeps contributing to ``u``; it just stops moving.
        """
        self._set_requires_grad(self.monoms[m], False)

    def unfreeze_mode(self, m):
        """Unfreeze the monoms of mode ``m`` -- a **row** of the grid."""
        self._set_requires_grad(self.monoms[m], True)

    def freeze_factor(self, k):
        """Freeze the monoms of factor ``k`` -- a **column** of the grid.

        All modes at once, for the ``k``-th factor. What alternating-direction
        minimisation cycles over: unfreeze one direction, optimise, move on.
        """
        self._set_requires_grad((row[k] for row in self.monoms), False)

    def unfreeze_factor(self, k):
        """Unfreeze the monoms of factor ``k`` -- a **column** of the grid."""
        self._set_requires_grad((row[k] for row in self.monoms), True)

    def freeze_monom(self, m, k):
        """Freeze the single monom ``w_m^k`` -- one **cell** of the grid.

        The finest granularity, for a classical ADM sweep that optimises one
        monom at a time.
        """
        self._set_requires_grad([self.monoms[m][k]], False)

    def unfreeze_monom(self, m, k):
        """Unfreeze the single monom ``w_m^k`` -- one **cell** of the grid."""
        self._set_requires_grad([self.monoms[m][k]], True)

    def add_mode(self):
        """Append one mode to the decomposition.

        Builds a new row of the grid: one ``TrainableField`` and one
        ``QuadratureAssembly`` per ``MonomSpec``. The new monoms bind to their
        factor's existing ``FactorSpace``, so they reuse its already-computed
        ``QuadratureContext`` rather than recomputing any geometry.

        Nothing is capped and nothing is frozen: the new monoms are born
        trainable, and the freeze state of the other modes is untouched. Deciding
        when to stop, and what to freeze, is the trainer's job.

        Two things the caller must do afterwards, for as long as no trainer does
        it for them: re-run :meth:`register_into` so the new fields reach the
        ``FieldLayout`` (it is idempotent), and hand the new parameters to the
        optimizer (see
        :meth:`neurom.neurom_model.NeuROMModel.add_mode_to_optimizer`).

        The new mode keeps its ``MonomSpec.init_values`` seed rather than being zeroed:
        an all-zero mode is a stationary point of the energy (every gradient
        component is proportional to the *other* factor, so both stay locked at
        0), which never takes off under a gradient optimizer. A non-zero
        parametric seed lets the linear load term drive the enrichment.

        Returns:
            int: Index of the mode just added.
        """
        m = self.n_modes
        fields = [
            TrainableField(
                name=f"{self.name}_dim{spec.space.name}_mode{m}",
                connectivity=spec.space.connectivity,
                init_values=spec.init_values,
                constraint=spec.constraint,
            )
            for spec in self.monom_specs
        ]
        self.monoms.append(nn.ModuleList(fields))
        self._assemblies.append(
            nn.ModuleList(
                [
                    QuadratureAssembly(spec.space.context, spec.sf, field)
                    for spec, field in zip(self.monom_specs, fields)
                ]
            )
        )
        return m

    def mode_parameters(self, m=None):
        """Return mode ``m``'s trainable monom parameters (optimizer-agnostic).

        The decomposition owns which tensors make up a mode; wiring them into an
        optimizer is :meth:`neurom.neurom_model.NeuROMModel.add_mode_to_optimizer`'s
        job, so the PGD stays agnostic to the optimizer.

        Args:
            m (int, optional): Index of the mode. Supports negative indexing
                (Python-style). Defaults to the last mode added
                (``n_modes - 1``).

        Returns:
            list[torch.Tensor]: The monom parameters of mode ``m``, one per factor.

        Raises:
            IndexError: If ``m`` is out of range for the existing modes.
        """
        n_modes = self.n_modes
        if m is None:
            m = n_modes - 1
        if m < 0:
            m += n_modes
        if not 0 <= m < n_modes:
            raise IndexError(f"Mode index {m} out of range for {n_modes} mode(s).")
        return [f.values_reduced for f in self.monoms[m]]

    def register_into(self, field_layout):
        """Bring ``field_layout`` in phase with the monoms that exist now.

        Idempotent: fields already registered are skipped
        (:meth:`neurom.field_layout.FieldLayout.add` is identity-aware), so this
        is called once at setup **and again after every** :meth:`add_mode`. That
        is how new monoms reach the layout without the decomposition ever
        holding a reference to it.
        """
        for mode in self.monoms:
            for field in mode:
                field_layout.add(field)

    def directory(self):
        """Ordered lookup table of the monom field names, keyed by factor.

        Returns:
            dict[str, list[str]]: factor name -> monom field names, one per mode
            (index ``m``). Feed to a physics/energy term to read each monom
            out of the FieldLayout by name (``field_layout[name]``). Grows after
            :meth:`add_mode`.
        """
        n = self.n_modes
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
        n = self.n_modes
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
        n_modes = self.n_modes
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
