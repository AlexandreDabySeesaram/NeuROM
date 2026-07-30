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


def pin_axis(exponents, axis=0, power=1):
    """Force one axis' column of an exponent set to a fixed power.

    A *transform*, not a third builder, so it composes with both
    :func:`uniform_exponents` and :func:`total_degree_exponents` rather than
    duplicating either. The motivating case is the "frozen support" schedule:
    with the space monom held fixed as a mere support for the parametric
    factors, its exponent should stay at 1 and the non-linearity should live in
    the parameters alone -- ``pin_axis(uniform_exponents(5, 3), 0, 1)`` turns
    ``{(2,2,2,2,2), (3,3,3,3,3)}`` into ``{(1,2,2,2,2), (1,3,3,3,3)}``.

    Pinning can collapse distinct rows onto each other and can manufacture the
    leading term (``total_degree``'s ``(2,1,1,1,1)`` pins to ``(1,1,1,1,1)``), so
    the result is deduplicated and the leading term dropped. Row order is
    otherwise preserved, first occurrence winning.

    Args:
        exponents (torch.Tensor): ``(T, d)`` integer exponent set.
        axis (int): Which column to pin, in the decomposition's axis order.
        power (int): The value to pin it to; must be ``>= 1``, since a zero
            exponent cannot satisfy the axis' essential boundary condition.

    Returns:
        torch.Tensor: ``(T', d)`` long tensor with ``T' <= T``.

    Raises:
        ValueError: if ``power < 1``, if ``axis`` is out of range, or if pinning
            leaves the set empty (every row was the leading term).
    """
    if not torch.is_tensor(exponents):
        exponents = torch.as_tensor(exponents)
    if exponents.dim() != 2:
        raise ValueError(
            f"exponents must be a (T, d) tensor; got shape {tuple(exponents.shape)}."
        )
    n_axes = exponents.shape[1]
    if not -n_axes <= axis < n_axes:
        raise ValueError(f"axis {axis} out of range for {n_axes} axes.")
    if power < 1:
        raise ValueError(
            f"power must be >= 1 (a zero exponent makes the term constant along "
            f"that axis, which cannot satisfy its essential boundary condition); "
            f"got {power}."
        )

    pinned = exponents.clone().to(torch.long)
    pinned[:, axis] = power

    kept, seen = [], set()
    for row in pinned:
        key = tuple(int(v) for v in row)
        if key in seen or all(v == 1 for v in key):
            continue
        seen.add(key)
        kept.append(key)
    if not kept:
        raise ValueError(
            f"pin_axis(..., axis={axis}, power={power}) left the exponent set "
            "empty: every row pinned onto the leading term (1, ..., 1), which is "
            "carried separately. Raise the bound of the underlying set."
        )
    return torch.tensor(kept, dtype=torch.long)


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

    Optional leading coefficient ``c_i``
        With ``leading_coefficients=True`` the leading term's fixed weight 1
        becomes a trainable scalar per mode,

            u = sum_i ( c_i prod_j w_ij + sum_lambda C_{i lambda} prod_j w_ij^lambda_j ).

        **This adds no expressivity.** Rescaling ``w_ij -> s_j w_ij`` with
        ``prod_j s_j = c`` reproduces any ``c`` exactly -- it *is* the gauge
        direction the fixed weight pins (see below). It is worth having anyway,
        for two reasons that are about the parameterisation, not the model:

        * *Reachability.* A mode with no linear part at all (``c_i -> 0``) is
          only a limit point of the fixed-weight parameterisation -- monoms to 0
          with ``C_lambda`` to infinity. Here it is an ordinary interior point.
        * *Scale.* Releasing ``c_i`` lets :meth:`renormalise` normalise all
          ``d`` monoms instead of ``d - 1``, which puts ``c_i`` and every row of
          ``C_i`` on one scale. See :meth:`renormalise_mode`.

        Off by default, and then no parameter is created at all, so ``state_dict``
        is unchanged and checkpoints written without it still load.

    Gauge degeneracy (known, not corrected)
        The representation is invariant under the *per-axis* rescaling

            w_ij -> s_j w_ij,  C_{i lambda} -> C_{i lambda} prod_j s_j^(-lambda_j)

        **subject to ``prod_j s_j = 1``**. The constraint is there because the
        leading term carries a fixed unit coefficient: rescaling every axis
        freely would scale ``prod_j w_ij`` and change the field. So pinning that
        coefficient already gauge-fixes one direction, and ``d - 1`` flat
        directions per mode remain.

        With ``leading_coefficients=True`` the constraint lifts (``c_i`` absorbs
        ``prod_j s_j``, and the leading term's exponent row ``(1, ..., 1)`` obeys
        the same rule as every other), so the orbit is ``d``-dimensional per mode
        and :meth:`renormalise` imposes ``d`` conditions instead of ``d - 1``.

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
        leading_coefficients (bool): Make each mode's leading weight ``c_i`` a
            trainable scalar instead of a fixed 1. Defaults to False, which
            creates no parameter and leaves ``state_dict`` untouched. See the
            ``Optional leading coefficient`` section above.
    """

    def __init__(
        self,
        axes,
        n_modes_max,
        exponents,
        name="poly_pgd",
        n_modes_ini=1,
        leading_coefficients=False,
        orthogonal_corrections=False,
    ):
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

        # The leading term's weight. Off by default, and then it is not a
        # parameter at all -- no ParameterList is built, so `state_dict` is
        # byte-for-byte what it was before this option existed and every
        # checkpoint written without it still loads. On, it is one scalar per
        # mode, initialised at 1 (the fixed weight it replaces) and frozen, so
        # even an enabled-but-unreleased decomposition evaluates identically.
        self.has_leading_coefficients = bool(leading_coefficients)
        if self.has_leading_coefficients:
            self.leading_coefficients = nn.ParameterList(
                [torch.nn.Parameter(torch.ones(())) for _ in range(self.n_modes_max)]
            )
            for c in self.leading_coefficients:
                c.requires_grad_(False)
        else:
            self.leading_coefficients = None

        # Deflate each correction against the leading term (see
        # `_deflation_axis`). No parameters, so `state_dict` is unchanged and
        # every checkpoint still loads -- but the *meaning* of a stored `C` moves,
        # so a checkpoint must be reloaded with the same flag it was written
        # with. Off by default, and then every code path below is bit-identical
        # to what it was before this option existed.
        self.orthogonal_corrections = bool(orthogonal_corrections)
        self._deflation_axes = None
        if self.orthogonal_corrections:
            self._deflation_axes = [
                self._deflation_axis(tuple(int(v) for v in lam))
                for lam in self.exponents
            ]

    @staticmethod
    def _deflation_axis(lam):
        """Which axis of exponent row ``lam`` carries the deflation.

        The inner product over a tensor-product domain factorises,
        ``<prod_j f_j, prod_j g_j> = prod_j <f_j, g_j>``, so **one** zero factor
        makes the whole product orthogonal to the leading term. Deflating a
        single axis is therefore enough -- and it is what keeps the cost finite:
        deflating every axis would expand one correction into ``2^d`` raw
        products, and the consuming loop is quadratic in the term count.

        The axis must have ``lam_j != 1``: for ``lam_j = 1`` the deflation is
        ``w_j - (<w_j,w_j>/<w_j,w_j>) w_j = 0`` and would annihilate the term.
        Such an axis always exists, because ``(1,) * d`` is rejected by
        :meth:`_validate_exponents` -- it *is* the leading term.

        Returns:
            int: the first axis with ``lam_j != 1``.
        """
        for j, power in enumerate(lam):
            if power != 1:
                return j
        raise ValueError(  # pragma: no cover -- _validate_exponents rejects it
            f"exponent row {lam} is the leading term; it cannot be deflated"
        )

    def deflation_factor(self, m, k, power):
        """``<w^p, w> / <w, w>`` on axis ``k`` of mode ``m``, by quadrature.

        The coefficient that makes ``w^p - beta w`` orthogonal to ``w``. Read
        off the *quadrature*, like :meth:`monom_norms` and for the same reason:
        a nodal inner product would make the result mesh-dependent.

        **Differentiable, deliberately** -- unlike ``monom_norms``, which runs
        under ``no_grad`` because the gauge fix is applied to the parameters
        rather than composed into the field. Here ``beta`` is part of the field
        the energy sees, so a gradient that ignored it would be the gradient of a
        different model.

        Args:
            m (int): mode index.
            k (int): axis index.
            power (int): the exponent ``p``.

        Returns:
            torch.Tensor: 0-dim.
        """
        r = self._assemblies[m][k].interpolate()
        numerator = integrate(r.u**power * r.u * r.measure)
        denominator = integrate(r.u * r.u * r.measure)
        return numerator / denominator

    def deflation_shrinkage(self, m, t):
        """How much deflation shrinks correction ``t`` of mode ``m``.

        ``||w_k^p - beta w_k|| / ||w_k||^p`` on the deflated axis, and exactly
        ``1.0`` when ``orthogonal_corrections`` is off.

        Exists for the diagnostics: a term magnitude read as
        ``|C| prod_j ||w_j||^lambda_j`` measures the *undeflated* product, so
        with deflation on it overstates the correction by this factor -- the
        component parallel to the leading term is the part that was removed.
        Multiplying by this restores the comparison without redefining the
        magnitude for the runs that do not use the flag.

        Returns:
            float
        """
        if not self.orthogonal_corrections:
            return 1.0
        k = self._deflation_axes[t]
        power = int(self.exponents[t, k])
        with torch.no_grad():
            r = self._assemblies[m][k].interpolate()
            beta = self.deflation_factor(m, k, power)
            deflated = r.u**power - beta * r.u
            numerator = integrate(deflated * deflated * r.measure).sqrt()
            plain = integrate(r.u * r.u * r.measure).sqrt() ** power
        return float(numerator / plain)

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

    def freeze_mode_leading_coefficient(self, m):
        """Freeze mode ``m``'s leading coefficient ``c_m``.

        Raises:
            RuntimeError: if the decomposition was built without
                ``leading_coefficients``, in which case the leading weight is a
                fixed 1 and there is nothing to freeze.
        """
        self._require_leading_coefficients()
        self.leading_coefficients[m].requires_grad_(False)

    def unfreeze_mode_leading_coefficient(self, m):
        """Unfreeze mode ``m``'s leading coefficient ``c_m``.

        See :meth:`freeze_mode_leading_coefficient`.
        """
        self._require_leading_coefficients()
        self.leading_coefficients[m].requires_grad_(True)

    def _require_leading_coefficients(self):
        if not self.has_leading_coefficients:
            raise RuntimeError(
                "this PolynomialNLPGD has no leading coefficients: the leading "
                "term's weight is a fixed 1. Build it with "
                "PolynomialNLPGD(..., leading_coefficients=True) to make it "
                "trainable."
            )

    def freeze_all(self):
        """Freeze every mode's monoms *and* coefficients, leading ones included."""
        super().freeze_all()
        # CPPGD.__init__ calls freeze_all() before this subclass has built its
        # coefficients; nothing to freeze on that first pass.
        if getattr(self, "coefficients", None) is None:
            return
        for m in range(self.n_modes_max):
            self.freeze_mode_coefficients(m)
        if getattr(self, "leading_coefficients", None) is not None:
            for m in range(self.n_modes_max):
                self.freeze_mode_leading_coefficient(m)

    def mode_coefficient_parameters(self, m=None):
        """Mode ``m``'s coefficient row alone, for a correction-only stage.

        Args:
            m (int, optional): Mode index, Python-style negative indexing
                supported. Defaults to the last-activated mode.

        Returns:
            list[torch.Tensor]: the ``(T,)`` coefficient parameter, preceded by
            the scalar leading coefficient when the decomposition has one.
        """
        m = self._resolve_mode(m)
        if self.has_leading_coefficients:
            return [self.leading_coefficients[m], self.coefficients[m]]
        return [self.coefficients[m]]

    def mode_parameters(self, m=None):
        """Mode ``m``'s monom parameters plus its coefficients.

        See :meth:`~neurom.decompositions.pgd.CPPGD.mode_parameters`; this adds
        the ``(T,)`` coefficient tensor, and the scalar leading coefficient
        before it when the decomposition has one, as the trailing entries.
        """
        return [*super().mode_parameters(m), *self.mode_coefficient_parameters(m)]

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

        Two regimes, according to whether the leading weight is a free parameter:

        * **Fixed leading weight** (the default). ``prod_j s_j = 1`` is forced,
          so only ``d - 1`` conditions are available: the last ``d - 1`` monoms
          go to unit norm and axis 0 absorbs the amplitude ``A``.
        * **Leading coefficient live.** The constraint lifts -- any overall
          scale can be absorbed by ``c_m`` -- so **all ``d``** monoms go to unit
          norm and ``c_m`` takes the amplitude. This is the regime worth having:
          with every ``||w_ij|| = 1``, every term's natural size
          ``prod_j ||w_ij||^lambda_j`` is 1 whatever ``lambda``, so ``c_m`` and
          every row of ``C_m`` sit on the *same* scale. Under the ``d - 1`` fix
          they sit at ``A^(1 - p)``, orders apart, which is what makes a single
          Adam ``coefficient_lr`` unable to serve them all.
        """
        norms = self.monom_norms(m)
        if not all(bool(torch.isfinite(n)) and float(n) > 0.0 for n in norms):
            return

        scales = [torch.ones_like(norms[0]) for _ in norms]
        if self.has_leading_coefficients:
            for k in range(len(norms)):
                scales[k] = 1.0 / norms[k]
        else:
            for k in range(1, len(norms)):
                scales[k] = 1.0 / norms[k]
                scales[0] = scales[0] * norms[k]

        with torch.no_grad():
            for k, s in enumerate(scales):
                self.monoms[m][k].values_reduced.mul_(s)
            if self.has_leading_coefficients:
                # The leading term's exponent row is (1, ..., 1), so it takes
                # the same prod_j s_j^(-lambda_j) rule as every other term.
                leading = torch.ones_like(norms[0])
                for s in scales:
                    leading = leading / s
                self.leading_coefficients[m] *= leading
            for t in range(self.n_terms):
                factor = torch.ones_like(norms[0])
                for k, s in enumerate(scales):
                    factor = factor * s ** (-int(self.exponents[t, k]))
                self.coefficients[m][t] *= factor

    # -- structure readback ----------------------------------------------------

    def term_is_inert(self, m, t):
        """Is correction term ``t`` of mode ``m`` both **frozen and exactly zero**?

        Such a term contributes nothing to the field and can never come to: it
        multiplies its product by ``0``, and with ``requires_grad=False`` it
        receives no gradient that could move it off ``0``. Assembling it is pure
        waste -- which is what :meth:`polynomial_directory`'s ``skip_inert``
        exists to avoid.

        **Both conditions are load-bearing.** A coefficient that is zero but
        *trainable* is the ordinary state of a fresh mode at the start of a
        ``joint`` stage: dropping it would remove it from the graph, so it would
        get no gradient and stay at zero forever -- a correction that silently
        never activates, which is the hardest failure mode to spot in this
        example. Testing ``== 0`` alone is therefore a bug, not an optimisation.

        Args:
            m (int): mode index.
            t (int): row of :attr:`exponents`.

        Returns:
            bool: True when the term can be dropped without changing anything.
        """
        coefficient = self.coefficients[m]
        # `requires_grad` lives on the whole row, so a frozen row may still hold
        # non-zero elements -- `staged` fits C and then freezes it. Hence the
        # per-element test after the per-row one.
        if coefficient.requires_grad:
            return False
        return float(coefficient[t].detach()) == 0.0

    def polynomial_directory(self, skip_inert=False):
        """Flattened enumeration of the decomposition's terms.

        The companion to the inherited :meth:`directory`, which still supplies
        the monom *names*. This supplies the *powers and weights*: with a
        polynomial mode the unit of summation is no longer the mode but the
        term, so a bilinear energy's double loop runs over pairs of these.

        Args:
            skip_inert (bool): drop the correction terms :meth:`term_is_inert`
                reports -- frozen at exactly zero, so contributing nothing and
                unable to start. Off by default, so the enumeration stays a
                faithful description of the decomposition unless a caller asks
                for the cheaper one. The consuming loop is quadratic, so during
                a linear phase (every ``C`` frozen at 0) this takes the cost from
                ``(n_modes * (1 + |I|))^2`` down to ``n_modes^2`` -- a factor 36
                at ``|I| = 5``. The **field is unchanged either way**; only the
                assembly cost moves.

        Returns:
            list[tuple[int, tuple[int, ...], torch.Tensor | None]]: one entry per
            term, as ``(mode_index, exponents, coefficient)``.

            * ``exponents`` -- one power per axis, in ``self.axes`` order, all
              ``>= 1``.
            * ``coefficient`` -- for a correction term, the matching element of
              ``self.coefficients[mode]``. For a mode's **leading** term,
              ``None`` when the leading weight is a fixed 1 (the default), or
              ``self.leading_coefficients[mode]`` when the decomposition was
              built with ``leading_coefficients=True``. A consumer that reads it
              as ``1.0 if coefficient is None else coefficient`` therefore needs
              no change to support either.

            Ordered mode-major: for each active mode, its leading term
            (``exponents = (1,) * n_axes``) then its ``|I|`` correction terms.

        Note:
            The consuming double loop is quadratic in ``len(...)`` =
            ``n_modes * (1 + |I|)`` -- 5 modes with ``|I| = 5`` is 900 pairs
            where ``CPPGD`` has 25.
        """
        out = []
        for m in range(self.n_modes_truncated):
            for lam, coefficient in self._term_rows(m, skip_inert=skip_inert):
                out.append((m, lam, coefficient))
        return out

    def _term_rows(self, m, skip_inert=False):
        """One mode's terms as ``(exponents, coefficient)`` pairs.

        The single place the term structure is spelled out, so
        :meth:`polynomial_directory` (which the energy reads) and
        :meth:`_mode_from_columns` (which ``evaluate`` and ``assemble`` read)
        cannot drift apart -- with ``orthogonal_corrections`` on they would
        otherwise describe two different fields, and the run would train one and
        report the other.

        Args:
            m (int): mode index.
            skip_inert (bool): see :meth:`polynomial_directory`.

        Returns:
            list[tuple[tuple[int, ...], torch.Tensor | None]]
        """
        leading = (1,) * len(self.axes)
        c = self.leading_coefficients[m] if self.has_leading_coefficients else None
        # The leading term is never skipped: with a fixed unit weight it has no
        # coefficient to be zero, and with `c` live a zero `c` is a state the
        # optimizer may be passing through rather than parked at.
        rows = [(leading, c)]
        for t, lam in enumerate(self.exponents):
            if skip_inert and self.term_is_inert(m, t):
                continue
            lam = tuple(int(v) for v in lam)
            coefficient = self.coefficients[m][t]
            if not self.orthogonal_corrections:
                rows.append((lam, coefficient))
                continue
            # `w_j^p -> w_j^p - beta w_j` on one axis. Deflation is *linear*, so
            # the deflated product expands into two ordinary products -- which is
            # what lets every consumer stay a plain "coefficient times a product
            # of powers" loop, at the price of one extra row per correction.
            k = self._deflation_axes[t]
            beta = self.deflation_factor(m, k, lam[k])
            shadow = lam[:k] + (1,) + lam[k + 1 :]
            rows.append((lam, coefficient))
            rows.append((shadow, -beta * coefficient))
        return rows

    # -- evaluation ------------------------------------------------------------

    def _mode_from_columns(self, cols, m):
        """Combine one mode's per-axis columns into the mode's value.

        Args:
            cols (list[torch.Tensor]): one interpolated column per axis, already
                broadcast-compatible with each other.
            m (int): mode index, selecting the coefficient row.

        Returns:
            torch.Tensor: ``c_m prod_k cols[k] + sum_t C[m, t] prod_k
            cols[k]**lam[t, k]``, with ``c_m`` a fixed 1 unless the
            decomposition carries leading coefficients.
        """
        total = None
        for lam, coefficient in self._term_rows(m):
            term = cols[0] ** lam[0]
            for k, c in enumerate(cols[1:], start=1):
                term = term * c ** lam[k]
            if coefficient is not None:
                term = coefficient * term
            total = term if total is None else total + term
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
