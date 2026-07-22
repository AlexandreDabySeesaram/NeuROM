"""Correctness tests for the 5-parametric (x, E1, E2, alpha, n) beam example.

The example script lives under docs/examples/ and its filename starts with a digit,
so it cannot be imported by module name; it is loaded from its path instead.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPO_ROOT
    / "docs"
    / "examples"
    / "1d_5-parametric_beam_PGD"
    / "1d_5-parametric_beam_deflection_PGD.py"
)


def load_module(path=EXAMPLE_PATH):
    """Import the example script from its path under an arbitrary module name."""
    spec = importlib.util.spec_from_file_location("beam5p", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def beam5p():
    return load_module()


def test_axes_are_the_five_expected_coordinates(beam5p):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    assert [axis.name for axis in problem.pgd.axes] == [
        "space",
        "E1",
        "E2",
        "alpha",
        "n",
    ]
    assert list(problem.pgd.directory()) == ["space", "E1", "E2", "alpha", "n"]
    assert problem.pgd.n_modes_max == 10
    assert problem.pgd.n_modes_truncated == 1


@pytest.mark.parametrize(
    "name, lo, hi, n_nodes",
    [
        ("space", 0.0, 10.0, 30),
        ("E1", 10.0, 100.0, 20),
        ("E2", 10.0, 100.0, 20),
        ("alpha", 2.0, 8.0, 15),
        ("n", 0.5, 5.0, 15),
    ],
)
def test_axis_intervals_and_resolutions(beam5p, name, lo, hi, n_nodes):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    positions = problem.axes[name].nodes_positions.values
    assert positions.shape == (n_nodes, 1)
    assert positions.min().item() == pytest.approx(lo)
    assert positions.max().item() == pytest.approx(hi)


def test_forward_interpolates_every_monom_and_the_load(beam5p):
    problem = beam5p.build_problem(lambda layout, decomposition: torch.zeros(()))

    layout = problem.model()

    # MidPoint1D: one quadrature point per element, n_nodes - 1 elements.
    expected = {"space": 29, "E1": 19, "E2": 19, "alpha": 14, "n": 14}
    directory = problem.pgd.directory()
    for axis_name, n_elements in expected.items():
        result = layout[directory[axis_name][0]]
        assert result.u.shape == (n_elements, 1, 1)
        assert result.x.shape == (n_elements, 1, 1)
        assert result.measure.shape == (n_elements, 1, 1)

    # The load must be sampled on the *same* quadrature as the space monom,
    # otherwise inner(load, X) silently broadcasts wrong.
    assert layout["load"].u.shape == layout[directory["space"][0]].u.shape
    assert torch.allclose(layout["load"].u, torch.full_like(layout["load"].u, 1000.0))


def test_n_nodes_override_builds_a_smaller_problem(beam5p):
    problem = beam5p.build_problem(
        lambda layout, decomposition: torch.zeros(()),
        n_nodes={"space": 5, "E1": 4, "E2": 4, "alpha": 4, "n": 4},
    )

    layout = problem.model()
    directory = problem.pgd.directory()
    assert layout[directory["space"][0]].u.shape == (4, 1, 1)
    assert layout[directory["n"][0]].u.shape == (3, 1, 1)


def test_loss_is_the_injected_callable(beam5p):
    seen = {}

    def spy(layout, decomposition):
        seen["layout"] = layout
        seen["decomposition"] = decomposition
        return torch.tensor(42.0)

    problem = beam5p.build_problem(spy)
    out = problem.model()

    assert problem.model.loss(out).item() == 42.0
    assert seen["layout"] is problem.field_layout
    assert seen["decomposition"] is problem.pgd
