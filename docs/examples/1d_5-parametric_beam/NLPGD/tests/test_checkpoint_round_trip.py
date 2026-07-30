"""Checkpointing a real CPPGD, on the 5-parametric beam.

The unit tests pin the mechanism against a stub. What only the real
decomposition can answer is whether its truncation survives a round trip: the
``active`` flags are buffers on ``QuadratureAssembly``, which is an
implementation detail of the library, not of the checkpoint format. If that ever
stops being true, a reloaded model would quietly evaluate mode 0 alone and every
figure drawn from a checkpoint would be wrong while looking plausible.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from neurom.training import (
    FixedIterations,
    GreedyTrainer,
    MaxStages,
    load_checkpoint,
    save_checkpoint,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE = EXAMPLE_DIR / "1d_5-parametric_beam_deflection_NLPGD.py"

TINY = {"space": 5, "E1": 4, "E2": 6, "alpha": 7, "n": 3}


def load_module():
    spec = importlib.util.spec_from_file_location("beam5nl", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def beam5p():
    return load_module()


def build(beam5p):
    torch.manual_seed(0)
    return beam5p.build_problem(beam5p.energy, n_modes_max=3, n_nodes=TINY)


def query_points(beam5p, n=7):
    """A handful of (x, E1, E2, alpha, n) tuples, in AXIS_ORDER."""
    columns = [
        torch.linspace(*beam5p.AXIS_BOUNDS[name], n) for name in beam5p.AXIS_ORDER
    ]
    return torch.stack(columns, dim=1)


def test_a_reloaded_decomposition_evaluates_identically(beam5p, tmp_path):
    problem = build(beam5p)
    trainer = GreedyTrainer(
        problem.model,
        stage_criterion=FixedIterations(10),
        enrichment_criterion=MaxStages(3),
    )
    history = trainer.enrich()
    points = query_points(beam5p)
    trained = problem.pgd.evaluate(points).clone()

    save_checkpoint(tmp_path / "run.pt", problem.model, history)

    fresh = build(beam5p)
    assert fresh.pgd.n_modes_truncated == 1  # nothing trained yet
    loaded_history, _ = load_checkpoint(tmp_path / "run.pt", fresh.model)

    # Same truncation, same values, same answer -- bitwise: it is the same
    # tensor written and read back, not a re-fit.
    assert fresh.pgd.n_modes_truncated == 3
    assert torch.equal(fresh.pgd.evaluate(points), trained)
    assert [r.energy for r in loaded_history.stages] == [
        r.energy for r in history.stages
    ]


def test_the_checkpoint_path_is_one_file_per_strategy(beam5p):

    paths = {
        name: beam5p.checkpoint_path(cls) for name, cls in beam5p.STRATEGIES.items()
    }

    # Five strategies compared against the same reference must not overwrite
    # each other's trained model -- the joint/staged pair above all, since those
    # two differ ONLY in schedule and would otherwise share a file.
    assert len(set(paths.values())) == len(beam5p.STRATEGIES)
    assert paths["joint"].name == "nlpgd5_jointnlgreedy.pt"
    assert paths["staged"].name == "nlpgd5_stagednlgreedy.pt"
    assert paths["greedy"].name == "nlpgd5_greedy.pt"
    assert paths["simultaneous"].name == "nlpgd5_simultaneous.pt"
    # And must not collide with the CP example's, which shares PARAM_SWEEP_DIR's
    # basename one directory over.
    assert paths["joint"].parent.parent.name == "NLPGD"


def test_a_leading_coefficient_survives_a_round_trip(beam5p, tmp_path):
    """``c_i`` is a parameter like any other -- but only when it exists.

    ``leading_coefficients`` adds state_dict keys, so a checkpoint written with
    it and reloaded into a problem built without it must *fail loudly* on the
    missing keys rather than reload a decomposition whose leading weights have
    silently reverted to 1. That failure mode would be invisible in the figures:
    the modes are all still there, only mis-weighted.
    """
    torch.manual_seed(0)
    problem = beam5p.build_problem(
        beam5p.energy, n_modes_max=3, n_nodes=TINY, leading_coefficients=True
    )
    history = beam5p.SupportNLGreedyTrainer(
        problem.model,
        stage_criterion=FixedIterations(10),
        enrichment_criterion=MaxStages(2),
        leading_coefficient=True,
    ).enrich()

    coefficients = [c.detach().clone() for c in problem.pgd.leading_coefficients]
    assert any(float(c) != 1.0 for c in coefficients), "nothing to round-trip"
    query = query_points(beam5p)
    before = problem.pgd.evaluate(query).detach().clone()

    path = tmp_path / "with_c.pt"
    save_checkpoint(path, problem.model, history)

    torch.manual_seed(1)  # a different seed, so a no-op load cannot pass
    reloaded = beam5p.build_problem(
        beam5p.energy, n_modes_max=3, n_nodes=TINY, leading_coefficients=True
    )
    load_checkpoint(path, reloaded.model)

    assert torch.equal(reloaded.pgd.evaluate(query), before)
    for got, want in zip(reloaded.pgd.leading_coefficients, coefficients):
        assert torch.equal(got.detach(), want)

    # ... and the mismatch raises rather than silently dropping c.
    with pytest.raises(RuntimeError, match="leading_coefficients"):
        load_checkpoint(path, build(beam5p).model)
