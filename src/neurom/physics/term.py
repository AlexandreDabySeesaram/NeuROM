from abc import ABC, abstractmethod


class Term(ABC):
    """General abstraction of a term in an expression

    A term provides an integrand() method which represents the integrand: function times measure, :math: `f dx`.
    This computation of the integrand is done late, i.e. only when integrand() method is done explicitly.
    """

    @abstractmethod
    def integrand(self, fields_layout):
        pass

    def __add__(self, other):
        return SumTerm([self, other])

    def __sub__(self, other):
        return SumTerm([self, -other])

    def __neg__(self):
        return NegTerm(self)


class SumTerm(Term):
    def __init__(self, terms):
        self.terms = []
        for t in terms:
            if isinstance(t, SumTerm):
                self.terms.extend(t.terms)
            else:
                self.terms.append(t)

    def integrand(self, fields_layout):
        expr = 0
        for t in self.terms:
            expr = expr + t.integrand(fields_layout)
        return expr


class NegTerm(Term):
    def __init__(self, term):
        self.term = term

    def integrand(self, fields_layout):
        return -self.term.integrand(fields_layout)
