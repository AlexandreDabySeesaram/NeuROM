"""Tests for the hyperparameter sweep tooling of the 5-parametric beam.

Loads the example script and ``sweep.py`` by path (they are scripts, not an
installed package), mirroring the sibling example tests.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "1d_5-parametric_beam_deflection_NLPGD.py"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load(EXAMPLE, "beam5nl")
SWEEP = EXAMPLE.parent / "sweep.py"
sw = _load(SWEEP, "beam5nl_sweep")


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


def test_build_optimizer_factory_gives_the_coefficients_their_own_lr():
    """The split that keeps the ``staged`` schedule from diverging.

    Not cosmetic: a shared ``lr`` moves ``C`` by ~0.1 on its first Adam step,
    where the target scale is ``1e-6`` or smaller. See
    ``build_optimizer_factory``'s measurement table.
    """
    cfg = ex.RunConfig(name="x", lr=0.05, coefficient_lr=1e-7)
    problem = ex.build_problem(
        lambda layout, decomposition: torch.zeros(()),
        n_modes_max=2,
        n_nodes={"space": 5, "E1": 4, "E2": 4, "alpha": 4, "n": 4},
    )
    problem.pgd.unfreeze_mode_coefficients(0)
    factory = ex.build_optimizer_factory(cfg, problem.pgd)

    opt = factory([p for p in problem.model.parameters() if p.requires_grad])

    by_lr = {group["lr"]: group["params"] for group in opt.param_groups}
    assert set(by_lr) == {0.05, 1e-7}
    assert by_lr[1e-7] == [problem.pgd.coefficients[0]]
    # The five monoms of the one active mode, and nothing else.
    assert len(by_lr[0.05]) == 5


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


def _row(energy, n_nodes=None):
    return {"config": {"n_nodes": n_nodes}, "result": {"final_energy": energy}}


def test_energy_band_is_centred_on_the_median_not_the_mean():
    # Four healthy rows and one blowup: a mean would be dragged to -1.2e11 and
    # would put every healthy row outside the band.
    rows = [_row(-2.0e11), _row(-2.1e11), _row(-2.05e11), _row(-2.02e11),
            _row(2.0e11)]
    lo, hi = sw.energy_bands(rows)[None]
    assert lo <= -2.05e11 <= hi
    assert all(sw._energy_flag(r, {None: (lo, hi)}) == " " for r in rows[:4])
    assert sw._energy_flag(rows[4], {None: (lo, hi)}) == "!"


def test_energy_band_flags_both_sides():
    rows = [_row(-2.0e11), _row(-2.05e11), _row(-2.1e11)]
    bands = sw.energy_bands(rows)
    assert sw._energy_flag(_row(-3.6e11), bands) == "!"  # over-minimised
    assert sw._energy_flag(_row(-2.6e10), bands) == "!"  # never got going


def test_energy_band_is_per_mesh():
    rows = [_row(-2.0e11), _row(-2.05e11), _row(-2.1e11),
            _row(-8.0e11, 400), _row(-8.1e11, 400), _row(-8.2e11, 400)]
    bands = sw.energy_bands(rows)
    assert set(bands) == {None, 400}
    assert sw._energy_flag(_row(-8.1e11, 400), bands) == " "
    assert sw._energy_flag(_row(-8.1e11), bands) == "!"


def test_energy_band_needs_three_rows_and_nan_is_never_in_band():
    assert sw.energy_bands([_row(-2.0e11), _row(-2.1e11)]) == {}
    assert sw._energy_flag(_row(-2.0e11), {}) == "?"
    bands = sw.energy_bands([_row(-2.0e11), _row(-2.05e11), _row(-2.1e11),
                             _row(float("nan"))])
    assert sw._energy_flag(_row(float("nan")), bands) == "!"
    assert sw._energy_flag({"config": {}, "result": {}}, bands) == "!"


def test_run_sweep_records_a_rank_curve_and_show_rank_curve_reads_it(
    tmp_path, monkeypatch, capsys
):
    """The curve must have one entry per trained rank, ending at the row's own
    numbers -- that last identity is what makes the curve and the summary
    columns readable as the same measurement."""
    # Error that improves with rank, so the stub cannot pass by returning a
    # constant: keyed off how many modes are active at call time.
    def fake_errors(pgd, reference=None):
        rank = int(pgd.n_modes_truncated)
        return {
            "overall": 1.0 / rank,
            "per_point": {"pA": 0.5 / rank, "pB": 2.0 / rank},
            "u_pgd": None,
        }

    monkeypatch.setattr(sw._example(), "relative_errors", fake_errors)
    monkeypatch.setattr(sw._example(), "load_reference_bundle", lambda: None)

    ledger = tmp_path / "res.jsonl"
    rows = sw.run_sweep([_tiny("curve")], ledger=ledger, verbose=False, plot=False,
                        checkpoint_dir=tmp_path)
    res = rows[0]["result"]
    curve = res["rank_curve"]

    assert [p["rank"] for p in curve] == list(range(1, res["n_modes"] + 1))
    assert curve[0]["overall"] == pytest.approx(1.0)
    assert curve[0]["worst"] == pytest.approx(2.0)
    # The last point restates the summary columns.
    assert curve[-1]["overall"] == pytest.approx(res["overall_error"])
    assert curve[-1]["worst"] == pytest.approx(res["worst_point_error"])
    # Truncation restored: the run is still at full rank afterwards.
    assert res["overall_error"] == pytest.approx(1.0 / res["n_modes"])

    sw.show_rank_curve(rows[0]["config_id"], ledger=ledger)
    out = capsys.readouterr().out
    assert "rank" in out and "worst" in out


def test_show_rank_curve_says_so_when_the_row_predates_it(tmp_path, capsys):
    ledger = tmp_path / "res.jsonl"
    sw.upsert_row(ledger, {"config_id": "old1", "config": {"name": "n"},
                           "result": {"overall_error": 0.1}})
    sw.show_rank_curve("old1", ledger=ledger)
    assert "no rank_curve" in capsys.readouterr().out
