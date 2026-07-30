"""The univariate family a polynomial decomposition's exponents index.

``PolynomialNLPGD`` writes a mode as ``c_i prod_j w_ij + sum_lambda C_i,lambda
prod_j psi_{lambda_j}(w_ij)``. What ``psi_p`` *is* was hard-coded as ``v ** p``
until this module; injecting it instead is what lets the exponent rows index an
**orthogonal** family without touching the term bookkeeping, the freeze API or
the energy's double loop.

Why it matters, measured. With ``psi_p(v) = v ** p`` and the monoms free,
``prod_j w_ij^p`` spans exactly the same rank-1 set as ``prod_j w_ij`` -- take
``w -> w^(1/p)`` -- so a lone correction *replaces* the leading term instead of
complementing it, and the mode falls back to CP expressed with a tiny
coefficient against a huge basis. On the 5-parametric beam that cost the leading
term 98-100% of its mode.

An orthogonal family removes it outright, and does so for **every** pair of
terms at once, because the quadrature inner product over a tensor-product domain
factorises::

    <prod_j psi_{a_j}(w_j), prod_j psi_{b_j}(w_j)>
        = prod_j <psi_{a_j}(w_j), psi_{b_j}(w_j)>

so two terms are orthogonal as soon as they differ on a **single** axis. The
leading term is ``psi_1 = identity``, a member of the same family, so
leading-vs-correction and correction-vs-correction overlap fall together.

Two caveats, both structural rather than tuning:

* **Orthogonality here is approximate.** ``<psi_a(w), psi_b(w)>`` integrates over
  the axis, which by change of variables is ``int psi_a psi_b rho`` with ``rho``
  the *law of w's values*. Legendre is orthogonal for the uniform law, not for
  that one. The residual overlap is therefore a number to measure, not to assume
  -- see the example's term diagnostics.
* **A non-monomial family does not vanish at zero** (``P_2(0) = -1/2``), so on an
  axis carrying a homogeneous Dirichlet condition it would break the boundary
  condition that ``v ** p`` inherits for free. Keep such axes on
  :class:`MonomialBasis`; one orthogonal axis already orthogonalises the product.
"""

import torch


class TermBasis:
    """One univariate family ``psi_p``, plus its derivative.

    Subclasses supply :meth:`value` and :meth:`derivative` for the integer powers
    an exponent row can carry (``>= 1``; see ``PolynomialNLPGD._validate_exponents``).

    Attributes:
        needs_normalised_argument (bool): whether ``psi_p`` is only meaningful on
            a bounded argument, so the caller must feed it ``w / ||w||_inf``
            rather than ``w``. False for the monomials, which are homogeneous and
            defined everywhere.
    """

    needs_normalised_argument = False

    def normalises(self, power):
        """Does degree ``power`` want ``w / ||w||_inf`` rather than ``w``?

        **Degree 1 never does**, even on a family that otherwise requires it, and
        this is the difference between a working decomposition and a broken one
        rather than a nicety.

        Why it is safe: ``psi_1`` is the identity, so ``psi_1(w) = ||w||_inf
        psi_1(w_hat)`` -- a *positive constant* multiple. Orthogonality against
        every other degree is therefore untouched, ``<w, psi_p(w_hat)> =
        ||w||_inf <w_hat, psi_p(w_hat)>``.

        Why it is necessary: ``w_hat`` is scale-invariant, so normalising degree
        1 quotients the leading term's amplitude away. The mode then has **one**
        multiplicative amplitude channel instead of ``d``, and Adam's per-stage
        step budget cannot cover it. Measured on the 5-parametric beam, rank 1,
        300 iterations: normalising degree 1 gave a stage-0 energy of -6.64e10
        against the monomials' -1.90e11, with the mode amplitude 5.5e4 against
        3.1e5. Each axis only has to reach ``(3e5)^(1/5) = 12.6`` when all five
        carry amplitude; alone, space would need ~756 and roughly 2400
        iterations to get there.
        """
        return self.needs_normalised_argument and power != 1

    def gauge_exponent(self, power):
        """How this factor scales when its monom is rescaled: ``s ** ?``.

        What ``renormalise`` must undo in the matching coefficient. ``power`` for
        a homogeneous family; for a normalised one, ``1`` at degree 1 (the raw
        monom, which does scale) and ``0`` above (``psi_p(w_hat)`` does not move
        at all).
        """
        if not self.normalises(power):
            return power
        return 0

    def value(self, power, v):
        """``psi_power(v)``, elementwise."""
        raise NotImplementedError

    def derivative(self, power, v):
        """``psi'_power(v)``, elementwise -- the derivative **in the argument**.

        The chain rule factor for the argument's own derivative (``dv/dx``, and
        the ``1/||w||_inf`` of a normalised argument) belongs to the caller, not
        here.
        """
        raise NotImplementedError


class MonomialBasis(TermBasis):
    """``psi_p(v) = v ** p`` -- the family every run before this module used.

    The default everywhere, so a decomposition built without an explicit basis is
    bit-identical to what it was, down to the ``state_dict`` keys.

    Homogeneous (``psi_p(s v) = s^p psi_p(v)``), which is what makes
    ``PolynomialNLPGD.renormalise`` able to absorb a monom rescaling into the
    coefficients -- the gauge fix's whole mechanism. It is also what makes the
    family *redundant*: see the module docstring.
    """

    needs_normalised_argument = False

    def value(self, power, v):
        return v**power

    def derivative(self, power, v):
        # `power * v ** (power - 1)`, and at power 1 that is `1 * v ** 0`, which
        # torch evaluates to ones of v's shape -- the path `pin_space_exponent`
        # exercises.
        return power * v ** (power - 1)


class LegendreBasis(TermBasis):
    """Legendre polynomials ``P_p``, evaluated at the monom's value.

    ``P_1(v) = v``, so the leading term is unchanged and the corrections join the
    same family. Written out rather than recursed: the exponent rows in use go to
    degree 4, and an explicit table keeps ``derivative`` exact instead of
    differentiating a recurrence.

    Orthogonal for the uniform measure on ``[-1, 1]``, hence
    ``needs_normalised_argument``: the caller must map the monom into that
    interval, or ``P_p`` is evaluated where it grows without bound and the
    family's orthogonality claim is void.

    **Not homogeneous, and not zero at zero.** Both matter:

    * ``P_p(s v) != s^p P_p(v)``, so a coefficient cannot absorb a monom
      rescaling. With the argument normalised, though, ``P_p(w/||w||_inf)`` is
      scale-*invariant*, so the correction simply does not move -- which is why
      ``renormalise`` must rescale the leading coefficient alone.
    * ``P_2(0) = -1/2``, so this family must not be used on an axis with a
      homogeneous Dirichlet condition.
    """

    needs_normalised_argument = True

    #: ``P_p`` and ``P_p'`` for the degrees the exponent builders can produce.
    _P = {
        1: (lambda v: v, lambda v: torch.ones_like(v)),
        2: (lambda v: 0.5 * (3.0 * v**2 - 1.0), lambda v: 3.0 * v),
        3: (lambda v: 0.5 * (5.0 * v**3 - 3.0 * v), lambda v: 0.5 * (15.0 * v**2 - 3.0)),
        4: (
            lambda v: 0.125 * (35.0 * v**4 - 30.0 * v**2 + 3.0),
            lambda v: 0.5 * (35.0 * v**3 - 15.0 * v),
        ),
    }

    def _row(self, power):
        try:
            return self._P[int(power)]
        except KeyError:
            raise ValueError(
                f"LegendreBasis has no degree {power}; it is tabulated for "
                f"{sorted(self._P)}. Add the row rather than falling back to a "
                "recurrence, so the derivative stays exact."
            ) from None

    def value(self, power, v):
        return self._row(power)[0](v)

    def derivative(self, power, v):
        return self._row(power)[1](v)
