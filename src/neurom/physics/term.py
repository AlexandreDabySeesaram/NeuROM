from abc import ABC, abstractmethod

from neurom.field_layout import FieldLayout


class Term(ABC):
    """Abstract base class for a term in a variational expression.

    Sub‑classes must implement :meth:`integrand` which receives a
    :class:`~neurom.field_layout.FieldLayout` and returns the value of the
    integrand (typically a ``torch.Tensor``), i.e.  :math: `f(x) dx`.

    The class provides operator overloads so that terms can be combined using
    ``+``, ``-`` and unary ``-``.  These overloads return :class:`SumTerm` or
    :class:`NegTerm` objects that preserve the original terms.
    """

    @abstractmethod
    def integrand(self, field_layout: FieldLayout):
        """Compute the integrand for the given field layout.

        Args:
            field_layout (FieldLayout): The :class:`~neurom.field_layout.FieldLayout` that provides access to interpolated field values.

        Returns:
            The evaluated integrand (e.g. a ``torch.Tensor``).  The return type
            is not fixed; any object supporting addition and multiplication by
            scalars can be used, as long as it is compatible with the rest of
            the expression.
        """
        raise NotImplementedError

    def __add__(self, other: "Term") -> "SumTerm":
        """Return a ``SumTerm`` representing ``self + other``.

        Args:
            other (Term): Another :class:`Term` to be added.

        Returns:
            A new :class:`SumTerm` containing ``self`` and ``other`` (or their
            flattened equivalents if they are already ``SumTerm`` instances).
        """
        return SumTerm([self, other])

    def __sub__(self, other: "Term") -> "SumTerm":
        """Return a ``SumTerm`` representing ``self - other``.

        The ``other`` term is first negated (via ``-other``) and then combined.

        Args:
            other (Term): The term to subtract from ``self``.

        Returns:
            A new :class:`SumTerm` containing ``self`` and the negated ``other``.
        """
        return SumTerm([self, -other])

    def __neg__(self) -> "NegTerm":
        """Return a ``NegTerm`` representing ``-self``.

        Returns:
            A :class:`NegTerm` that will negate the result of ``self.integrand``.
        """
        return NegTerm(self)


class SumTerm(Term):
    """Composite term representing the sum of multiple sub‑terms.

    ``SumTerm`` flattens any nested ``SumTerm`` instances so that the internal
    ``terms`` list always contains only concrete (non‑sum) ``Term`` objects.
    This simplifies later processing and keeps the expression tree shallow.

    Attributes:
        terms (list[Term]): The flat list of sub‑terms that will be summed.
    """

    def __init__(self, terms: list[Term]):
        """Create a ``SumTerm`` from an iterable of terms.

        Args:
            terms (list[Term]): A sequence of :class:`Term` objects.  If any element is
                itself a ``SumTerm``, its internal ``terms`` are unpacked
                (flattened) into the new instance.

        Raises:
            TypeError: If an element of ``terms`` is not a ``Term`` instance.
        """
        self.terms: list[Term] = []
        for t in terms:
            if isinstance(t, SumTerm):
                self.terms.extend(t.terms)
            else:
                self.terms.append(t)

    def integrand(self, field_layout: FieldLayout):
        """Evaluate the sum of all sub‑terms' integrands.

        Args:
            field_layout (FieldLayout): The layout used by each sub‑term to compute its
                contribution.

        Returns:
            The element‑wise sum of the integrands of all contained terms.
        """
        expr = 0
        for t in self.terms:
            expr = expr + t.integrand(field_layout)
        return expr


class NegTerm(Term):
    """Wrapper term that negates the integrand of another term.

    Attributes:
        term (Term): The underlying term whose integrand will be negated.
    """

    def __init__(self, term: Term):
        """Create a ``NegTerm`` for the given term.

        Args:
            term (Term): The term to be negated.

        Raises:
            TypeError: If ``term`` is not an instance of :class:`Term`.
        """
        self.term: Term = term

    def integrand(self, field_layout: FieldLayout):
        """Negate the underlying term's integrand.

        Args:
            field_layout (FieldLayout): The layout passed to the wrapped term.

        Returns:
            The negated value of ``self.term.integrand(field_layout)``.
        """
        return -self.term.integrand(field_layout)
