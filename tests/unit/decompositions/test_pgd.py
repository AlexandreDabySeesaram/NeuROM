import pytest
import torch

from neurom.decompositions import Axis
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import NoConstraint

torch.set_default_dtype(torch.float32)


def make_axis(name="space", n=5, lo=0.0, hi=10.0):
    coords = torch.linspace(lo, hi, n).unsqueeze(-1)
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(nodes, elements)
    positions = Field(name=f"{name}_positions", topology=topology, values=coords)
    sf = LinearSegment()
    return Axis(
        name=name,
        nodes_positions=positions,
        sf=sf,
        mapping=IsoparametricMapping1D(sf),
        quad=TwoPoints1D(),
        constraint=NoConstraint(),
        init_values=torch.zeros(n, 1),
    )


def test_axis_exposes_topology_from_positions():
    axis = make_axis()
    assert axis.name == "space"
    # topology property must be the SAME object as the positions' topology
    assert axis.topology is axis.nodes_positions.topology
    assert axis.topology.n_nodes == 5
