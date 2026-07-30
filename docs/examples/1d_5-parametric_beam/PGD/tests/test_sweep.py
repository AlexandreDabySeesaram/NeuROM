"""Tests for the hyperparameter sweep tooling of the 5-parametric beam.

Loads the example script and ``sweep.py`` by path (they are scripts, not an
installed package), mirroring the sibling example tests.
"""

import importlib.util
from pathlib import Path

import torch

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "1d_5-parametric_beam_deflection_PGD.py"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load(EXAMPLE, "beam5")
SWEEP = EXAMPLE.parent / "sweep.py"
sw = _load(SWEEP, "beam5_sweep")


def test_config_id_is_stable_and_order_independent():
    a = ex.RunConfig(name="x", stage_tol=1e-6, min_iter=200)
    b = ex.RunConfig(name="x", min_iter=200, stage_tol=1e-6)
    assert ex.config_id(a) == ex.config_id(b)


def test_config_id_changes_when_a_knob_changes():
    a = ex.RunConfig(name="x")
    b = ex.RunConfig(name="x", stage_tol=1e-6)
    assert ex.config_id(a) != ex.config_id(b)


def test_config_id_is_hex8():
    cid = ex.config_id(ex.RunConfig(name="x"))
    assert len(cid) == 8 and all(c in "0123456789abcdef" for c in cid)


def test_build_criteria_maps_fields():
    cfg = ex.RunConfig(
        name="x", stage_tol=1e-6, window=10, max_iter=400, min_iter=200,
        stage_floor=2.0, enrichment_tol=5e-4, enrichment_floor=3.0,
    )
    stage, enrich = ex.build_criteria(cfg)
    assert (stage.tol, stage.window, stage.max_iter, stage.min_iter, stage.floor) == (
        1e-6, 10, 400, 200, 2.0
    )
    assert (enrich.tol, enrich.floor) == (5e-4, 3.0)


def test_build_optimizer_factory_uses_lr():
    cfg = ex.RunConfig(name="x", lr=0.05)
    factory = ex.build_optimizer_factory(cfg)
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = factory(p)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.05


def test_load_ledger_missing_file_is_empty(tmp_path):
    assert sw.load_ledger(tmp_path / "none.jsonl") == []


def test_upsert_appends_then_replaces(tmp_path):
    ledger = tmp_path / "l.jsonl"
    sw.upsert_row(ledger, {"config_id": "aaaa", "result": {"overall_error": 0.2}})
    sw.upsert_row(ledger, {"config_id": "bbbb", "result": {"overall_error": 0.1}})
    assert len(sw.load_ledger(ledger)) == 2
    # same id replaces, does not duplicate
    sw.upsert_row(ledger, {"config_id": "aaaa", "result": {"overall_error": 0.05}})
    rows = sw.load_ledger(ledger)
    assert len(rows) == 2
    a = next(r for r in rows if r["config_id"] == "aaaa")
    assert a["result"]["overall_error"] == 0.05


def _tiny(name, **over):
    base = dict(
        name=name, strategy="greedy",
        stage_tol=1e-3, window=5, max_iter=8, min_iter=4,
        enrichment_tol=1e-1, n_modes_max=2,
        n_nodes={"space": 6, "E1": 4, "E2": 4, "alpha": 4, "n": 4},
    )
    base.update(over)
    return ex.RunConfig(**base)


def test_run_sweep_writes_one_row_per_config_and_caches(tmp_path, monkeypatch):
    # avoid needing the FEM reference file: stub relative_errors on the exact
    # example-module instance the runner uses (memoised by sweep._example()).
    monkeypatch.setattr(
        sw._example(), "relative_errors",
        lambda pgd, reference=None: {
            "overall": 0.5, "per_point": {"pA": 0.5, "pB": 0.9}, "u_pgd": None,
        },
    )
    ledger = tmp_path / "res.jsonl"
    configs = [_tiny("a"), _tiny("b", stage_tol=1e-4)]

    rows = sw.run_sweep(configs, ledger=ledger, verbose=False, plot=False,
                        checkpoint_dir=tmp_path)
    assert len(rows) == 2
    ids = {r["config_id"] for r in rows}
    assert len(ids) == 2
    for r in rows:
        assert set(r["config"]) >= {"name", "strategy", "stage_tol", "n_nodes"}
        res = r["result"]
        assert res["overall_error"] == 0.5
        assert res["worst_point_error"] == 0.9
        assert res["worst_point_label"] == "pB"
        assert res["per_point_error"] == {"pA": 0.5, "pB": 0.9}
        assert res["n_modes"] >= 1
        assert res["n_stages"] >= 1
        assert isinstance(res["stages"], list) and res["stages"]
        assert {"amplitude", "max_correlation"} <= set(res["stages"][0])

    # re-run: cached, no new rows
    rows2 = sw.run_sweep(configs, ledger=ledger, verbose=False, plot=False,
                         checkpoint_dir=tmp_path)
    assert len(rows2) == 2

    # change a knob on "a": a third row, first two untouched
    configs2 = [_tiny("a", min_iter=5)] + configs
    rows3 = sw.run_sweep(configs2, ledger=ledger, verbose=False, plot=False,
                         checkpoint_dir=tmp_path)
    assert len(rows3) == 3
