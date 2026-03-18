import pytest
import torch

from neurom.physics.term import Term, SumTerm, NegTerm
from neurom.field_layout import FieldLayout

torch.set_default_dtype(torch.float32)


class DummyTerm(Term):
    """Simple concrete Term returning a constant tensor.

    The ``value`` attribute is returned as the integrand result.
    """

    def __init__(self, value):
        self.value = torch.tensor(value, dtype=torch.get_default_dtype())

    def integrand(self, field_layout: FieldLayout) -> torch.Tensor:  # noqa: D401
        """Return the stored constant tensor regardless of ``field_layout``.

        Args:
            field_layout: The layout object passed by the term infrastructure.
                It is unused in this dummy implementation.

        Returns:
            The constant tensor stored in ``self.value``.
        """
        return self.value


class TestTerm:
    """Test suite for the :mod:`neurom.physics.term` module.

    The class groups all functional tests as methods and provides a shared
    ``relative_tolerance`` attribute used for ``pytest.approx`` comparisons.
    """

    #: Relative tolerance used for all numeric approximations.
    relative_tolerance = 1e-6

    def test_addition_creates_sumterm_and_integrates(self):
        """Test that the ``+`` operator creates a :class:`SumTerm`.

        The test verifies:
        * The result is an instance of ``SumTerm``.
        * Its ``terms`` attribute contains the two original ``DummyTerm`` objects.
        * The combined integrand equals the element‑wise sum of the two tensors.
        """
        t1 = DummyTerm(2.0)
        t2 = DummyTerm(3.0)
        sum_term = t1 + t2

        assert isinstance(sum_term, SumTerm)
        # The SumTerm should contain exactly the two original terms.
        assert sum_term.terms == [t1, t2]
        # The integrand should be the sum of the two tensors.
        assert sum_term.integrand(None).item() == pytest.approx(
            5.0, rel=self.relative_tolerance
        )

    def test_subtraction_creates_sumterm_with_negated_term(self):
        """Test that the ``-`` operator creates a ``SumTerm`` with a ``NegTerm``.

        Steps:
        1. Subtract ``t2`` from ``t1``.
        2. Ensure the result is a ``SumTerm``.
        3. Verify the second term is a ``NegTerm`` wrapping ``t2``.
        4. Check that the integrand equals ``t1 - t2``.
        """
        t1 = DummyTerm(5.0)
        t2 = DummyTerm(1.5)
        result = t1 - t2

        assert isinstance(result, SumTerm)
        # ``result.terms`` should contain ``t1`` and a ``NegTerm`` wrapping ``t2``.
        assert result.terms[0] is t1
        assert isinstance(result.terms[1], NegTerm)
        assert result.terms[1].term is t2
        # The integrand should be 5.0 - 1.5 = 3.5.
        assert result.integrand(None).item() == pytest.approx(
            3.5, rel=self.relative_tolerance
        )

    def test_negation_returns_negterm_and_integrates(self):
        """Test that the unary ``-`` operator returns a ``NegTerm``.

        The test confirms that the ``NegTerm`` wraps the original term and that its
        integrand is the negated tensor.
        """
        t = DummyTerm(4.0)
        neg = -t

        assert isinstance(neg, NegTerm)
        assert neg.term is t
        assert neg.integrand(None).item() == pytest.approx(
            -4.0, rel=self.relative_tolerance
        )

    def test_sumterm_flattening_of_nested_sumterms(self):
        """Test that nested ``SumTerm`` objects are flattened.

        By chaining ``+`` operations, nested ``SumTerm`` instances are created.
        The constructor should flatten them so that ``terms`` contains a flat list
        of the original operands.  The integrand is then verified to be the total
        sum of all constituent tensors.
        """
        # Create nested SumTerms via successive additions.
        t1 = DummyTerm(1.0)
        t2 = DummyTerm(2.0)
        t3 = DummyTerm(3.0)
        nested = (t1 + t2) + t3

        # The resulting object should be a SumTerm with three flat terms.
        assert isinstance(nested, SumTerm)
        assert nested.terms == [t1, t2, t3]
        # Verify the integrand sums all values.
        assert nested.integrand(None).item() == pytest.approx(
            6.0, rel=self.relative_tolerance
        )
