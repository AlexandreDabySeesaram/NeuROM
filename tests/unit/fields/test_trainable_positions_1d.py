import torch
import torch.nn.functional as F

from neurom.meshes import Topology
from neurom.fields import TrainablePositions1D
from neurom.fields.trainable_positions_1d import inv_softplus

torch.set_default_dtype(torch.float32)


def make_topology(n):
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    return Topology(nodes, elements)


def test_inv_softplus_is_inverse_of_softplus():
    y = torch.tensor([0.01, 0.1, 0.5, 1.0, 5.0])
    x = inv_softplus(y)
    assert torch.allclose(F.softplus(x), y, atol=1e-6)


def test_full_values_reproduces_uniform_mesh():
    n = 11
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert torch.allclose(pos.full_values(), x, atol=1e-5)


def test_full_values_reproduces_nonuniform_mesh():
    n = 6
    x = torch.tensor([0.0, 0.05, 0.2, 0.5, 0.85, 1.0]).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert torch.allclose(pos.full_values(), x, atol=1e-5)


def test_coordinates_shape_is_n_minus_one():
    n = 8
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    assert pos.coordinates.shape == (n - 1,)
    assert pos.coordinates.requires_grad


def test_endpoints_fixed_and_strictly_increasing_for_arbitrary_params():
    n = 8
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    with torch.no_grad():
        pos.coordinates.add_(torch.randn_like(pos.coordinates))
    fv = pos.full_values().reshape(-1)
    assert fv[0].item() == 0.0
    assert torch.isclose(fv[-1], torch.tensor(1.0), atol=1e-6)
    assert bool((fv[1:] > fv[:-1]).all())


def test_gradients_flow_to_coordinates():
    n = 7
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    pos = TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    loss = (pos.full_values() ** 2).sum()
    loss.backward()
    assert pos.coordinates.grad is not None
    assert torch.isfinite(pos.coordinates.grad).all()


def test_at_elements_matches_gather_and_dim():
    n = 5
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    top = make_topology(n)
    pos = TrainablePositions1D(name="p", topology=top, initial_positions=x)
    ae = pos.at_elements()
    assert ae.shape == (n - 1, 2, 1)
    assert torch.allclose(ae, pos.full_values()[top.connectivity])
    assert pos.dim == 1


def test_rejects_non_increasing_initial_positions():
    n = 4
    x = torch.tensor([0.0, 0.5, 0.4, 1.0]).reshape(-1, 1)
    try:
        TrainablePositions1D(name="p", topology=make_topology(n), initial_positions=x)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-increasing initial positions")
