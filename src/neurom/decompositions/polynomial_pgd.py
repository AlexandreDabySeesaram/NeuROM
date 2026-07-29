import itertools

import torch
import torch.nn as nn

from neurom.constraints.no_constraint import NoConstraint
from neurom.decompositions.pgd import CPPGD
from neurom.integrate import integrate
from neurom.interpolation.point_wise_interpolator import PointWiseInterpolator


def _is_homogeneous(constraint):
    """Whether rescaling a field's free DOFs rescales the whole field.

    True when the constraint imposes nothing (``NoConstraint``) or imposes only
    zeros; false when it pins a DOF to a non-zero value, since that DOF would
    not follow the rescaling.

    A false here is worth reading as a warning about the *model*, not just about
    :func:`PolynomialNLPGD.renormalise`: a non-zero imposed value is not
    correctly imposed by a separated representation in the first place. See the
    ``Non-zero Dirichlet`` note on :class:`~neurom.decompositions.pgd.CPPGD`.
    """
    if isinstance(constraint, NoConstraint):
        return True
    imposed = getattr(constraint, "values_imposed", None)
    if imposed is None:
        return True
    return bool(torch.all(imposed == 0))


def uniform_exponents(n_axes, max_power):
    """Exponent set ``I = {(p, ..., p) : p in 2..max_power}``.

    The plan's simple case: every axis carries the same power, so the set has
    ``max_power - 1`` rows whatever the number of axes. ``p = 1`` is excluded
    because ``(1, ..., 1)`` is the leading term of the decomposition, which
    :class:`PolynomialNLPGD` carries separately with a fixed unit coefficient.

    Args:
        n_axes (int): Number of axes ``d``.
        max_power (int): Largest power ``n``; must be ``>= 2`` to be non-empty.

    Returns:
        torch.Tensor: ``(T, d)`` long tensor of exponent vectors.
    """
    if max_power < 2:
        raise ValueError(
            f"uniform_exponents needs max_power >= 2 (p = 1 is the leading term, "
            f"carried separately); got {max_power}."
        )
    return torch.tensor(
        [[p] * n_axes for p in range(2, max_power + 1)], dtype=torch.long
    )


def total_degree_exponents(n_axes, max_total):
    """Exponent set ``I = {lambda : lambda_j >= 1, sum_j lambda_j <= max_total}``.

    Zero exponents are forbidden: ``w^0 = 1`` would make a term constant along
    that axis, which does not satisfy an essential boundary condition carried by
    the axis' monoms (a constant cannot meet ``u(0) = 0``). Requiring
    ``lambda_j >= 1`` everywhere makes every term inherit the monoms' constraints
    for free -- at the price that the set is empty unless ``max_total > n_axes``.

    ``(1, ..., 1)`` is excluded, as in :func:`uniform_exponents`.

    Args:
        n_axes (int): Number of axes ``d``.
        max_total (int): Bound ``p`` on the total degree; must be ``> n_axes``.

    Returns:
        torch.Tensor: ``(T, d)`` long tensor of exponent vectors, lexicographic.
    """
    if max_total <= n_axes:
        raise ValueError(
            f"total_degree_exponents({n_axes}, {max_total}) is empty: every "
            f"exponent must be >= 1, so the total degree is at least "
            f"n_axes = {n_axes}, and the only vector of that degree is the "
            f"leading term (1, ..., 1). Need max_total > {n_axes}."
        )
    rows = [
        lam
        for lam in itertools.product(range(1, max_total - n_axes + 2), repeat=n_axes)
        if n_axes < sum(lam) <= max_total
    ]
    return torch.tensor(rows, dtype=torch.long)


class PolynomialNLPGD(CPPGD):
    """Non-linear PGD whose modes are polynomials in their own monoms.

    Extends :class:`~neurom.decompositions.pgd.CPPGD` from the multilinear
    ``u = sum_i prod_j w_ij`` to

        u = sum_i ( prod_j w_ij  +  sum_{lambda in I} C_{i lambda} prod_j w_ij^lambda_j )

    where ``I`` is an exponent set (see :func:`uniform_exponents`,
    :func:`total_degree_exponents`) and ``C`` is a trainable coefficient row per
    mode.

    Separability is preserved
        Each term ``prod_j w_ij^lambda_j`` is still a product over axes, so a
        separated energy keeps factorising into one 1-D moment per axis; the
        moment merely becomes ``int w_ij^lambda_j w_i'j^lambda'_j J``. Nothing
        new is assembled: powers are elementwise on the interpolated values and
        ``C`` never reaches a ``QuadratureAssembly``. The monom grid, the
        assemblies and the greedy lifecycle are inherited unchanged.

    Coefficient lifecycle -- ``requires_grad`` only
        Unlike the monoms, the coefficients carry no ``active`` flag. They do not
        need one: ``C`` is zero-initialised, and

            d/dw_ij [ C_{i lambda} prod_j w_ij^lambda_j ]
                = C_{i lambda} lambda_j w_ij^(lambda_j - 1) (...)

        vanishes with ``C``. A frozen ``C = 0`` therefore contributes neither to
        the energy nor to the monom gradients -- it is numerically
        indistinguishable from the polynomial terms being absent, at the cost of
        some dead autograd nodes. That makes the staged protocol simply

            add_mode()                        # new mode, C frozen at 0
            ... train ...                     # behaves as pure CP-PGD
            unfreeze_mode_coefficients(m)
            ... train ...                     # trains the polynomial correction

        :meth:`add_mode` is inherited untouched and never releases coefficients;
        that is always an explicit call.

    Gauge degeneracy (known, not corrected)
        The representation is invariant under the *per-axis* rescaling

            w_ij -> s_j w_ij,  C_{i lambda} -> C_{i lambda} prod_j s_j^(-lambda_j)

        **subject to ``prod_j s_j = 1``**. The constraint is there because the
        leading term carries a fixed unit coefficient: rescaling every axis
        freely would scale ``prod_j w_ij`` and change the field. So pinning that
        coefficient already gauge-fixes one direction, and ``d - 1`` flat
        directions per mode remain.

        Consequence: the Hessian is singular in ``d - 1`` directions per mode,
        and the monom/coefficient split is not identifiable. :meth:`renormalise`
        fixes the gauge by imposing exactly those ``d - 1`` conditions; see there.

    Writing an energy against this class
        Read the monom *names* from the inherited :meth:`directory` and the
        *powers and weights* from :meth:`polynomial_directory`; the bilinear
        double loop runs over term pairs, not mode pairs. On a differentiated
        axis the chain rule gives ``grad(X^lambda) = lambda X^(lambda-1) grad X``,
        so the powered gradient must be built from **both** ``result.u`` and
        ``jacobian_field`` -- not from ``jacobian_field`` alone.

    Args:
        axes (list[Axis]): The ordered axes; all must be scalar (``dim == 1``).
        n_modes_max (int): Maximum number of modes.
        exponents (torch.Tensor): ``(T, n_axes)`` integer exponent set ``I``.
            Every entry must be ``>= 1``, rows must be distinct, and no row may
            equal ``(1, ..., 1)``.
        name (str): Prefix for the monom field names.
        n_modes_ini (int): Number of initially active (trainable) modes.
    """

    def __init__(self, axes, n_modes_max, exponents, name="poly_pgd", n_modes_ini=1):
        axes = list(axes)
        for a in axes:
            if a.init_values.shape[1] > 1:
                raise ValueError(
                    f"PolynomialNLPGD requires scalar axes: axis '{a.name}' has "
                    f"dim {a.init_values.shape[1]}. A power of a vector-valued "
                    "monom has no defined meaning here."
                )
        super().__init__(
            axes=axes, n_modes_max=n_modes_max, name=name, n_modes_ini=n_modes_ini
        )

        exponents = self._validate_exponents(exponents, len(axes))
        # Buffer, not a plain attribute, so the exponent set rides along in
        # state_dict -- same idiom as QuadratureAssembly.active.
        self.register_buffer("exponents", exponents)

        # One (T,) coefficient row per mode, parallel to `self.monoms`. A
        # ParameterList rather than a single (n_modes, T) tensor because
        # requires_grad is per-mode and cannot be set on a slice.
        self.coefficients = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(exponents.shape[0]))
                for _ in range(self.n_modes_max)
            ]
        )
        # Every row starts frozen at 0: a freshly built decomposition behaves as
        # exactly CPPGD until a coefficient row is explicitly released.
        for c in self.coefficients:
            c.requires_grad_(False)

    @staticmethod
    def _validate_exponents(exponents, n_axes):
        if not torch.is_tensor(exponents):
            exponents = torch.as_tensor(exponents)
        if exponents.dim() != 2 or exponents.shape[1] != n_axes:
            raise ValueError(
                f"exponents must be a (T, n_axes) tensor with n_axes = {n_axes}; "
                f"got shape {tuple(exponents.shape)}."
            )
        if exponents.dtype not in (torch.int32, torch.int64, torch.int16):
            raise ValueError(
                f"exponents must have an integer dtype; got {exponents.dtype}."
            )
        exponents = exponents.to(torch.long)
        if exponents.shape[0] == 0:
            raise ValueError("exponents is empty: the exponent set I has no rows.")
        if bool((exponents < 1).any()):
            raise ValueError(
                "every exponent must be >= 1: a zero exponent makes the term "
                "constant along that axis, which cannot satisfy the axis' "
                "essential boundary conditions."
            )
        if bool((exponents == 1).all(dim=1).any()):
            raise ValueError(
                "exponents must not contain (1, ..., 1): that is the leading "
                "term, carried separately with a fixed unit coefficient."
            )
        if torch.unique(exponents, dim=0).shape[0] != exponents.shape[0]:
            raise ValueError("exponents contains duplicate rows.")
        return exponents

    @property
    def n_terms(self) -> int:
        """Number of polynomial correction terms per mode, ``|I|``."""
        return int(self.exponents.shape[0])

    # -- coefficient lifecycle -------------------------------------------------

    def freeze_mode_coefficients(self, m):
        """Freeze mode ``m``'s coefficient row (the monoms are untouched)."""
        self.coefficients[m].requires_grad_(False)

    def unfreeze_mode_coefficients(self, m):
        """Unfreeze mode ``m``'s coefficient row (the monoms are untouched)."""
        self.coefficients[m].requires_grad_(True)

    def freeze_all(self):
        """Freeze every mode's monoms *and* coefficients."""
        super().freeze_all()
        # CPPGD.__init__ calls freeze_all() before this subclass has built its
        # coefficients; nothing to freeze on that first pass.
        if getattr(self, "coefficients", None) is None:
            return
        for m in range(self.n_modes_max):
            self.freeze_mode_coefficients(m)

    def mode_coefficient_parameters(self, m=None):
        """Mode ``m``'s coefficient row alone, for a correction-only stage.

        Args:
            m (int, optional): Mode index, Python-style negative indexing
                supported. Defaults to the last-activated mode.

        Returns:
            list[torch.Tensor]: a single-element list holding the ``(T,)``
            coefficient parameter.
        """
        return [self.coefficients[self._resolve_mode(m)]]

    def mode_parameters(self, m=None):
        """Mode ``m``'s monom parameters plus its coefficient row.

        See :meth:`~neurom.decompositions.pgd.CPPGD.mode_parameters`; this adds
        the ``(T,)`` coefficient tensor as the trailing entry.
        """
        return [*super().mode_parameters(m), self.coefficients[self._resolve_mode(m)]]

    def _resolve_mode(self, m):
        """Normalise a (possibly negative or ``None``) mode index against actives."""
        n_active = int(self.n_modes_truncated)
        if m is None:
            m = n_active - 1
        if m < 0:
            m += n_active
        if not 0 <= m < n_active:
            raise IndexError(
                f"Mode index {m} out of range for {n_active} active mode(s)."
            )
        return m

    # -- gauge fixing ----------------------------------------------------------

    def monom_norms(self, m):
        """Quadrature L2 norms ``sqrt(int w_ij^2 dx_j)`` of mode ``m``'s monoms.

        The *quadrature* norm, not ``values_reduced.norm()``: the nodal norm is
        mesh-dependent, so a gauge fix built on it would shift whenever a mesh
        changes.

        Returns:
            list[torch.Tensor]: one 0-dim tensor per axis, detached.
        """
        norms = []
        with torch.no_grad():
            for k in range(len(self.axes)):
                r = self._assemblies[m][k].interpolate()
                norms.append(integrate(r.u * r.u * r.measure).sqrt())
        return norms

    def renormalise(self):
        """Fix the gauge on every active mode, leaving the field unchanged.

        The representation has ``d - 1`` flat directions per mode (see the class
        docstring). This imposes exactly ``d - 1`` conditions: the **last**
        ``d - 1`` monoms are scaled to unit quadrature norm and the **first**
        axis absorbs the scale, so the amplitude ends up on axis 0 (for the beam
        examples, the space factor -- matching the ``seed_amplitude``
        convention) and the parametric factors carry pure shape.

        Concretely, per mode ``i``::

            s_j = 1 / ||w_ij||   for j >= 1
            s_0 = prod_{j>=1} ||w_ij||
            w_ij       -> s_j w_ij
            C_{i lambda} -> C_{i lambda} prod_j s_j^(-lambda_j)

        ``prod_j s_j = 1`` holds by construction, which is exactly the condition
        the gauge orbit requires -- so this is an *exact*, energy-preserving
        reparameterisation, not a regularisation. There is no hyperparameter and
        no bias on the minimiser.

        Applied to every active mode, including frozen ones: the transformation
        preserves the field, so a frozen mode's contribution is untouched even
        though its stored parameters change. A mode whose monoms are not all
        finite and non-zero is skipped -- an unseeded or collapsed monom has no
        scale to normalise.

        Call it at a stage boundary, not mid-stage: it rescales parameters, so
        any optimizer state referring to them (Adam moments) goes stale.
        :meth:`~neurom.training.base.PGDTrainer.fix_gauge` is the seam that does
        this, just before ``make_optimizer``; whether it runs is the trainer's
        ``renormalise`` flag, not the decomposition's business.

        WARNING -- the constraint check below is a symptom, not the disease.
        ``renormalise`` scales ``values_reduced``, but the field is
        ``constraint.expand(values_reduced, dofs_free)``, which writes the
        imposed DOFs from a buffer no scaling reaches: with a non-zero imposed
        value the gauge fix silently stops being field-preserving. Turning it
        off with ``renormalise=False`` makes the error go away but leaves the
        deeper problem, which is that a non-zero Dirichlet is not correctly
        imposed by *any* separated representation here -- see the ``Non-zero
        Dirichlet`` note on :class:`~neurom.decompositions.pgd.CPPGD`. Lift.

        Raises:
            ValueError: if any axis carries a non-homogeneous constraint.
                Rescaling a monom scales its free DOFs but leaves the imposed
                ones fixed, so a non-zero imposed value would change the field.
        """
        for a in self.axes:
            if not _is_homogeneous(a.constraint):
                raise ValueError(
                    f"renormalise() needs homogeneous constraints, but axis "
                    f"'{a.name}' imposes non-zero values. Rescaling a monom "
                    "scales its free DOFs but leaves the imposed ones fixed, so "
                    "the field would change. `PGDTrainer(..., renormalise=False)` "
                    "silences this, but a non-zero Dirichlet is not correctly "
                    "imposed by the separated representation either (every mode "
                    "gets the same imposed value): lift the BC instead."
                )
        for m in range(self.n_modes_truncated):
            self.renormalise_mode(m)

    def renormalise_mode(self, m):
        """Apply the gauge fix of :meth:`renormalise` to mode ``m`` alone.

        Does not re-check the constraints; :meth:`renormalise` is the guarded
        entry point.
        """
        norms = self.monom_norms(m)
        if not all(bool(torch.isfinite(n)) and float(n) > 0.0 for n in norms):
            return

        scales = [torch.ones_like(norms[0]) for _ in norms]
        for k in range(1, len(norms)):
            scales[k] = 1.0 / norms[k]
            scales[0] = scales[0] * norms[k]

        with torch.no_grad():
            for k, s in enumerate(scales):
                self.monoms[m][k].values_reduced.mul_(s)
            for t in range(self.n_terms):
                factor = torch.ones_like(norms[0])
                for k, s in enumerate(scales):
                    factor = factor * s ** (-int(self.exponents[t, k]))
                self.coefficients[m][t] *= factor

    # -- structure readback ----------------------------------------------------

    def polynomial_directory(self):
        """Flattened enumeration of the decomposition's terms.

        The companion to the inherited :meth:`directory`, which still supplies
        the monom *names*. This supplies the *powers and weights*: with a
        polynomial mode the unit of summation is no longer the mode but the
        term, so a bilinear energy's double loop runs over pairs of these.

        Returns:
            list[tuple[int, tuple[int, ...], torch.Tensor | None]]: one entry per
            term, as ``(mode_index, exponents, coefficient)``.

            * ``exponents`` -- one power per axis, in ``self.axes`` order, all
              ``>= 1``.
            * ``coefficient`` -- ``None`` for the leading term of each mode
              (fixed weight 1, not a parameter), otherwise the matching element
              of ``self.coefficients[mode]``.

            Ordered mode-major: for each active mode, its leading term
            (``exponents = (1,) * n_axes``) then its ``|I|`` correction terms.

        Note:
            The consuming double loop is quadratic in ``len(...)`` =
            ``n_modes * (1 + |I|)`` -- 5 modes with ``|I| = 5`` is 900 pairs
            where ``CPPGD`` has 25.
        """
        leading = (1,) * len(self.axes)
        rows = [tuple(int(v) for v in lam) for lam in self.exponents]
        out = []
        for m in range(self.n_modes_truncated):
            out.append((m, leading, None))
            for t, lam in enumerate(rows):
                out.append((m, lam, self.coefficients[m][t]))
        return out

    # -- evaluation ------------------------------------------------------------

    def _mode_from_columns(self, cols, m):
        """Combine one mode's per-axis columns into the mode's value.

        Args:
            cols (list[torch.Tensor]): one interpolated column per axis, already
                broadcast-compatible with each other.
            m (int): mode index, selecting the coefficient row.

        Returns:
            torch.Tensor: ``prod_k cols[k] + sum_t C[m, t] prod_k cols[k]**lam[t, k]``.
        """
        total = cols[0]
        for c in cols[1:]:
            total = total * c
        for t in range(self.n_terms):
            term = cols[0] ** int(self.exponents[t, 0])
            for k, c in enumerate(cols[1:], start=1):
                term = term * c ** int(self.exponents[t, k])
            total = total + self.coefficients[m][t] * term
        return total

    def evaluate(self, coords):
        """Evaluate ``u`` at matched query points (diagonal), summed over modes.

        Same contract as :meth:`~neurom.decompositions.pgd.CPPGD.evaluate` --
        ``(P, n_axes)`` in, ``(P, 1)`` out -- with each mode's polynomial
        correction included. Each monom is interpolated **once** and the cached
        column is raised to every power, rather than re-interpolating per term.
        """
        coords = self._as_axis_columns(coords)
        total = None
        for m in range(self.n_modes_truncated):
            cols = []
            for k, axis in enumerate(self.axes):
                pwi = PointWiseInterpolator(
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
                w = pwi.at_position(coords[k].reshape(-1))  # (P, 1, 1)
                cols.append(w.reshape(w.shape[0], -1))  # (P, 1)
            mode = self._mode_from_columns(cols, m)
            total = mode if total is None else total + mode
        return total

    def assemble(self, coords):
        """Assemble the full separated tensor over the per-axis coordinates.

        Same contract as :meth:`~neurom.decompositions.pgd.CPPGD.assemble`, but
        summing the leading term plus one weighted tensor product per exponent
        row. All axes are scalar here (enforced in ``__init__``), so the einsum
        of the CP case degenerates to a plain outer product and the trailing
        component dimension is always absent.
        """
        n_modes = self.n_modes_truncated
        per_axis = []  # per_axis[k]: (n_modes, N_k)
        for k, axis in enumerate(self.axes):
            cols = []
            for m in range(n_modes):
                pwi = PointWiseInterpolator(
                    axis.mesh, axis.sf, self.monoms[m][k], axis.mapping
                )
                cols.append(pwi.at_position(coords[k].reshape(-1)).reshape(-1))
            per_axis.append(torch.stack(cols, dim=0))

        total = None
        for m in range(n_modes):
            cols = [
                _broadcast_axis(arr[m], k, len(per_axis))
                for k, arr in enumerate(per_axis)
            ]
            mode = self._mode_from_columns(cols, m)
            total = mode if total is None else total + mode
        return total


def _broadcast_axis(column, k, n_axes):
    """Reshape a 1-D per-axis column to broadcast along grid axis ``k``.

    Turns ``(N_k,)`` into the shape with ``N_k`` in position ``k`` and 1
    elsewhere, so that a plain product of the reshaped columns is the tensor
    product over the coordinate grid.
    """
    shape = [1] * n_axes
    shape[k] = column.shape[0]
    return column.reshape(shape)
