import pytest
import torch

from neurom.constraints import NoConstraint
from neurom.decompositions import FactorSpace, MonomSpec
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Connectivity, Mesh
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearBar


@pytest.fixture
def make_factor_space():
    """Factory for a uniform 1-D ``FactorSpace`` on ``n`` nodes over ``[lo, hi]``.

    Isoparametric: the mapping is built on ``LinearBar``, the same shape
    function the factors posed on this space use by default.
    """

    def _make_factor_space(name="space", n=5, lo=0.0, hi=10.0):
        coords = torch.linspace(lo, hi, n).unsqueeze(-1)
        nodes = torch.arange(0, n)
        elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
        connectivity = Connectivity(nodes, elements)
        positions = Field(
            name=f"{name}_positions", connectivity=connectivity, values=coords
        )
        mesh = Mesh(connectivity, positions)
        return FactorSpace(
            name=name,
            mesh=mesh,
            mapping=IsoparametricMapping1D(LinearBar(), mesh),
            quad=TwoPoints1D(),
        )

    return _make_factor_space


@pytest.fixture
def make_spec(make_factor_space):
    """Factory for a CP-PGD monom blueprint on a fresh uniform 1-D space.

    ``dim`` is the width of the monom values on that factor: 1 for a scalar
    factor, >1 for the single vector-valued factor a CP decomposition admits.
    Pass ``space`` to post the spec on an existing ``FactorSpace`` instead of a
    fresh one -- that is how two specs come to share one space.
    """

    def _make_spec(name="space", n=5, lo=0.0, hi=10.0, dim=1, space=None):
        if space is None:
            space = make_factor_space(name=name, n=n, lo=lo, hi=hi)
        return MonomSpec(
            space=space,
            sf=LinearBar(),
            constraint=NoConstraint(),
            init_values=torch.zeros(space.connectivity.n_nodes, dim),
        )

    return _make_spec


@pytest.fixture
def two_specs(make_spec):
    """The standard 2-factor setup: space on [0, 10], Young modulus E on [100, 1000].

    Space nodes are [0, 2.5, 5, 7.5, 10] and E nodes [100, 400, 700, 1000] --
    the values the hand-computed oracles in the tests are built on.
    """
    return [
        make_spec(name="space", n=5, lo=0.0, hi=10.0),
        make_spec(name="E", n=4, lo=100.0, hi=1000.0),
    ]
