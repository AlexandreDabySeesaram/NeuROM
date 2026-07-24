"""Tests for the direct-FEM reference solutions of the 5-parametric bar.

The reference is the yardstick every decomposition strategy is measured against,
so it is the one thing in this example that must be checked against something
external -- here, the closed-form constant-modulus solution.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = EXAMPLE_DIR / "reference_fem_solution.py"


def load_module(path=REFERENCE_PATH, name="reference_fem_solution"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_module():
    return load_module()


def test_constant_modulus_matches_the_analytical_parabola(reference_module):
    """With E1 == E2 the graded bar collapses to the 2-parametric analytical case."""
    error = reference_module.check_against_analytical()
    assert error < 1e-2


def test_sampled_parameters_are_in_bounds_and_reproducible(reference_module):
    example = reference_module.load_example()
    first = reference_module.sample_parameters(n_points=7, seed=3)
    second = reference_module.sample_parameters(n_points=7, seed=3)

    assert first.shape == (7, 4)
    assert torch.equal(first, second)
    for column, name in enumerate(reference_module.PARAM_NAMES):
        low, high = example.AXIS_BOUNDS[name]
        assert first[:, column].min() >= low
        assert first[:, column].max() <= high


def test_solution_is_clamped_and_negative_under_a_positive_load(reference_module):
    """f > 0 with the ``elastic + int f u`` convention pushes u negative inside."""
    example = reference_module.load_example()
    x_samples = torch.linspace(example.X_MIN, example.X_MAX, 21)
    u, energy = reference_module.solve_one(
        {"E1": 20.0, "E2": 80.0, "alpha": 6.0, "n": 3.0}, x_samples, example=example
    )

    assert u[0].abs() < 1e-6
    assert u[-1].abs() < 1e-6
    assert u[1:-1].max() < 0.0
    assert energy < 0.0


def test_softer_zone_deflects_more(reference_module):
    """Halving both moduli must scale the deflection up (linear problem: exactly 2x)."""
    example = reference_module.load_example()
    x_samples = torch.linspace(example.X_MIN, example.X_MAX, 21)
    params = {"E1": 40.0, "E2": 80.0, "alpha": 5.0, "n": 2.0}
    stiff, _ = reference_module.solve_one(params, x_samples, example=example)
    soft, _ = reference_module.solve_one(
        {**params, "E1": 20.0, "E2": 40.0}, x_samples, example=example
    )

    assert soft[1:-1] == pytest.approx(2.0 * stiff[1:-1], rel=1e-3)


def test_the_shipped_bundle_labels_its_highlighted_points(reference_module):
    """The extreme points must be present, labelled, and flagged for plotting."""
    bundle = reference_module.load_reference()
    names = [name for name, _ in reference_module.EXTREME_POINTS]

    assert bundle["metadata"]["highlight"] == names
    assert set(names) <= set(bundle["labels"])
    assert len(bundle["labels"]) == bundle["params"].shape[0] == bundle["u"].shape[0]

    # The stored parameters of a highlighted point are the ones declared here.
    for name, point in reference_module.EXTREME_POINTS:
        row = bundle["params"][bundle["labels"].index(name)]
        stored = dict(zip(bundle["param_names"], (v.item() for v in row)))
        assert stored == pytest.approx(point)


def test_saved_reference_round_trips(reference_module, tmp_path):
    bundle = {
        "x": torch.linspace(0.0, 1.0, 5),
        "params": torch.zeros(2, 4),
        "param_names": list(reference_module.PARAM_NAMES),
        "u": torch.zeros(2, 5),
        "energy": torch.zeros(2),
        "metadata": {"n_nodes": 3},
    }
    path = reference_module.save_reference(bundle, tmp_path / "ref.pt")
    loaded = reference_module.load_reference(path)

    assert loaded["param_names"] == bundle["param_names"]
    assert torch.equal(loaded["x"], bundle["x"])


def test_missing_reference_says_how_to_generate_it(reference_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="reference_fem_solution.py"):
        reference_module.load_reference(tmp_path / "absent.pt")
