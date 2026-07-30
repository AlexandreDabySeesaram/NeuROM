"""Unit tests for the univariate families a polynomial decomposition indexes.

The families themselves are a few lines of arithmetic; what these tests are
really for is the two properties the rest of the code leans on -- that the
monomial family is *exactly* what the code did before it was injectable, and
that ``derivative`` really is the derivative of ``value``. The second is the one
that would fail silently: on the monomial path the normalising scale is 1, so a
dropped or doubled ``1/scale`` shows up only under Legendre, and only as a
slightly wrong energy gradient.
"""

import pytest
import torch

from neurom.decompositions import LegendreBasis, MonomialBasis

torch.set_default_dtype(torch.float64)

VALUES = torch.tensor([-0.9, -0.3, 0.0, 0.25, 0.8])


# --------------------------------------------------------------------------
# The families
# --------------------------------------------------------------------------


@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_monomial_is_the_plain_power(power):
    assert torch.equal(MonomialBasis().value(power, VALUES), VALUES**power)
    assert torch.equal(
        MonomialBasis().derivative(power, VALUES), power * VALUES ** (power - 1)
    )


def test_legendre_matches_the_written_out_polynomials():
    basis = LegendreBasis()
    v = VALUES
    assert torch.allclose(basis.value(1, v), v)
    assert torch.allclose(basis.value(2, v), (3 * v**2 - 1) / 2)
    assert torch.allclose(basis.value(3, v), (5 * v**3 - 3 * v) / 2)
    assert torch.allclose(basis.value(4, v), (35 * v**4 - 30 * v**2 + 3) / 8)


@pytest.mark.parametrize("basis", [MonomialBasis(), LegendreBasis()])
@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_derivative_is_the_derivative_of_value(basis, power):
    # Against autograd rather than a second hand-written table: two tables
    # written from the same source would agree on a shared mistake.
    v = VALUES.clone().requires_grad_(True)
    basis.value(power, v).sum().backward()

    assert torch.allclose(basis.derivative(power, VALUES), v.grad)


def test_legendre_refuses_a_degree_it_does_not_tabulate():
    with pytest.raises(ValueError, match="no degree 5"):
        LegendreBasis().value(5, VALUES)


# --------------------------------------------------------------------------
# The two properties the decomposition relies on
# --------------------------------------------------------------------------


def test_only_legendre_asks_for_a_normalised_argument():
    # The flag is what routes `basis_value` through `w / ||w||_inf`, and what
    # makes `renormalise_mode` skip an axis. Both branches key off it, so it is
    # worth pinning rather than reading off the class.
    assert not MonomialBasis().needs_normalised_argument
    assert LegendreBasis().needs_normalised_argument


def test_the_monomials_are_homogeneous_and_legendre_is_not():
    # Homogeneity is exactly what lets `renormalise` absorb a monom rescaling
    # into the coefficients: psi_p(s v) = s^p psi_p(v). Legendre breaks it, which
    # is why `renormalise_mode` must not apply the s^(-lambda) rule to such an
    # axis -- doing so would change the field rather than preserve it.
    s, p = 2.0, 3
    assert torch.allclose(
        MonomialBasis().value(p, s * VALUES), s**p * MonomialBasis().value(p, VALUES)
    )
    assert not torch.allclose(
        LegendreBasis().value(p, s * VALUES), s**p * LegendreBasis().value(p, VALUES)
    )


def test_legendre_does_not_vanish_at_zero():
    # The reason space keeps the monomial family: a space monom is zero at both
    # ends by its Dirichlet condition, `w ** p` inherits that, and P_2 does not.
    zero = torch.zeros(1)
    assert float(MonomialBasis().value(2, zero)) == 0.0
    assert float(LegendreBasis().value(2, zero)) == pytest.approx(-0.5)
