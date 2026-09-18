import pytest
import torch

from neurom.constraints import NoConstraint
from neurom.decompositions import Axis
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Connectivity, Mesh
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearBar


@pytest.fixture
def make_axis():
    """Factory for a uniform 1-D CP-PGD axis on ``n`` nodes over ``[lo, hi]``.

    ``dim`` is the width of the monom values on that axis: 1 for a scalar
    factor, >1 for the single vector-valued factor a CP decomposition admits.
    """

    def _make_axis(name="space", n=5, lo=0.0, hi=10.0, dim=1):
        coords = torch.linspace(lo, hi, n).unsqueeze(-1)
        nodes = torch.arange(0, n)
        elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
        connectivity = Connectivity(nodes, elements)
        positions = Field(
            name=f"{name}_positions", connectivity=connectivity, values=coords
        )
        sf = LinearBar()
        mesh = Mesh(connectivity, positions)
        return Axis(
            name=name,
            mesh=mesh,
            sf=sf,
            mapping=IsoparametricMapping1D(sf, mesh),
            quad=TwoPoints1D(),
            constraint=NoConstraint(),
            init_values=torch.zeros(n, dim),
        )

    return _make_axis


@pytest.fixture
def two_axes(make_axis):
    """The standard 2-axis setup: space on [0, 10], Young modulus E on [100, 1000].

    Space nodes are [0, 2.5, 5, 7.5, 10] and E nodes [100, 400, 700, 1000] --
    the values the hand-computed oracles in the tests are built on.
    """
    return [
        make_axis(name="space", n=5, lo=0.0, hi=10.0),
        make_axis(name="E", n=4, lo=100.0, hi=1000.0),
    ]
