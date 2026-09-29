import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from latentedge.cli import cli
from latentedge.sweep import (
    Scenario,
    SIGNAL_ALWAYS, SIGNAL_MODEL, SIGNAL_ORACLE, SIGNAL_SHUFFLED, WINDOW_TEST, WINDOW_VALIDATE,
    build_scenarios, predictions_for, select_on_validate, split_windows, write_sweep,
)
from tests.test_cli_backtest import _train


def test_scenarios_cover_the_grid_and_both_reference_strategies_on_both_windows():
    scenarios = build_scenarios([0.0, 0.001, 0.002], [0.002, 0.005], [0.1, 0.5])

    for window in (WINDOW_VALIDATE, WINDOW_TEST):
        in_window = [s for s in scenarios if s.window == window]
        assert len([s for s in in_window if s.signal == SIGNAL_MODEL]) == 6 + 2
        assert len([s for s in in_window if s.top_fraction is not None]) == 2
        assert len([s for s in in_window if s.signal == SIGNAL_ALWAYS]) == 1
        assert len([s for s in in_window if s.signal == SIGNAL_ORACLE]) == 1
    # the rule is chosen on validate, so it always runs before the test window
    assert scenarios[0].window == WINDOW_VALIDATE and scenarios[-1].window == WINDOW_TEST


def test_windows_are_the_recorded_test_start_and_the_validated_bars_just_before_it():
    bars = pd.DataFrame({"bar_start": range(100, 200)})

    windows = split_windows(bars, test_start=170, validate_bars=15)

    assert list(windows[WINDOW_TEST]["bar_start"]) == list(range(170, 200))
    assert list(windows[WINDOW_VALIDATE]["bar_start"]) == list(range(155, 170))


def test_the_validate_window_never_reaches_into_the_test_window_or_before_the_data():
    bars = pd.DataFrame({"bar_start": range(10)})
    windows = split_windows(bars, test_start=4, validate_bars=50)
    assert list(windows[WINDOW_VALIDATE]["bar_start"]) == [0, 1, 2, 3]


def _row(window, signal, min_edge, full, ret):
    return {"window": window, "signal": signal, "min_edge": min_edge, "full_size_return": full, "total_return_usd": ret}


def test_selection_picks_the_best_validate_rule_and_reports_its_test_result():
    rows = [
        _row(WINDOW_VALIDATE, SIGNAL_MODEL, 0.0, 0.002, -50.0),
        _row(WINDOW_VALIDATE, SIGNAL_MODEL, 0.001, 0.002, 20.0),
        _row(WINDOW_VALIDATE, SIGNAL_ORACLE, 0.0, 0.002, 900.0),  # references are never selected
        _row(WINDOW_TEST, SIGNAL_MODEL, 0.0, 0.002, 500.0),  # the best test row must not influence the choice
        _row(WINDOW_TEST, SIGNAL_MODEL, 0.001, 0.002, -7.0),
    ]

    best, test = select_on_validate(rows)

    assert best["min_edge"] == 0.001
    assert test["total_return_usd"] == -7.0


def test_sweeps_are_never_overwritten(tmp_path: Path):
    fixed = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)

    first = write_sweep(tmp_path, {"model": "a"}, [{"x": 1}], now=lambda: fixed)
    second = write_sweep(tmp_path, {"model": "b"}, [{"x": 2}], now=lambda: fixed)

    assert first != second
    assert json.loads(first.read_text())["model"] == "a"
    assert json.loads(second.read_text())["scenarios"] == [{"x": 2}]
    assert first.with_suffix(".csv").exists()


def test_sweep_command_replays_the_grid_and_saves_a_comparable_record(tmp_path: Path):
    swaps, model = _train(tmp_path)
    out_dir = tmp_path / "sweeps"

    result = CliRunner().invoke(
        cli,
        ["sweep", "--swaps", str(swaps), "--model", str(model), "--out-dir", str(out_dir),
         "--min-edges", "0,0.002", "--full-size-returns", "0.002", "--top-fractions", "0.1,1.0",
         "--shuffle-seeds", ""],
    )

    assert result.exit_code == 0, result.output
    (saved,) = list(out_dir.glob("*.json"))
    record = json.loads(saved.read_text())
    assert len(record["scenarios"]) == 2 * (2 + 2 + 2)  # (2 edge rules + 2 rank rules + 2 references) on 2 windows
    assert {r["window"] for r in record["scenarios"]} == {WINDOW_VALIDATE, WINDOW_TEST}
    assert record["model_sha256"] and record["test_start"] and record["min_edges"] == [0.0, 0.002]
    test_model_rows = [
        r for r in record["scenarios"]
        if r["window"] == WINDOW_TEST and r["signal"] == SIGNAL_MODEL and r["top_fraction"] is None
    ]
    assert test_model_rows[0]["bars"] == record["training_metrics"]["splits"]["test"]["n"]
    # a higher bar for taking a trade can only mean fewer trades
    assert test_model_rows[1]["num_trades"] <= test_model_rows[0]["num_trades"]
    # the oracle only takes bars that really were profitable, so it can't lose
    oracle = next(r for r in record["scenarios"] if r["window"] == WINDOW_TEST and r["signal"] == SIGNAL_ORACLE)
    assert oracle["total_return_usd"] >= 0
    assert "saved to" in result.output


def test_sweep_refuses_a_model_that_was_never_trained(tmp_path: Path):
    result = CliRunner().invoke(cli, ["sweep", "--model", str(tmp_path / "nope.safetensors"), "--swaps", str(tmp_path / "s.parquet")])
    assert result.exit_code != 0
    assert "train the model first" in result.output


def test_a_rank_rule_trades_the_top_share_even_when_no_prediction_is_above_zero():
    validate = np.linspace(-0.003, -0.001, 100)  # a model that never predicts a gain
    test = np.array([-0.0035, -0.002, -0.0011, -0.0005])
    scenario = Scenario(WINDOW_TEST, SIGNAL_MODEL, 0.0, 0.002, top_fraction=0.10)

    signal = predictions_for(scenario, test, validate, pd.DataFrame({"net_return": test}))

    # the cutoff is the validation window's 90th percentile (about -0.0012), not the test window's own
    assert list(signal) == [0.0, 0.0, 0.002, 0.002]


def test_a_rank_rule_of_everything_trades_every_bar():
    validate = np.linspace(-0.003, -0.001, 100)
    scenario = Scenario(WINDOW_TEST, SIGNAL_MODEL, 0.0, 0.002, top_fraction=1.0)

    signal = predictions_for(scenario, np.array([-0.5, -0.004]), validate, pd.DataFrame({"net_return": [0, 0]}))

    assert list(signal) == [0.002, 0.002]


def test_shuffled_scenarios_repeat_the_rank_rules_for_each_seed_except_trade_everything():
    scenarios = build_scenarios([0.0], [0.002], [0.1, 0.5, 1.0], shuffle_seeds=[0, 1])

    for window in (WINDOW_VALIDATE, WINDOW_TEST):
        shuffled = [s for s in scenarios if s.window == window and s.signal == SIGNAL_SHUFFLED]
        assert sorted((s.seed, s.top_fraction) for s in shuffled) == [(0, 0.1), (0, 0.5), (1, 0.1), (1, 0.5)]
    assert not [s for s in build_scenarios([0.0], [0.002], [0.1]) if s.signal == SIGNAL_SHUFFLED]


def test_a_shuffled_rule_trades_the_same_number_of_bars_as_the_model_but_not_the_same_bars():
    rng = np.random.default_rng(3)
    validate = rng.normal(size=500)
    test = rng.normal(size=400)
    frame = pd.DataFrame({"net_return": test})
    model = Scenario(WINDOW_TEST, SIGNAL_MODEL, 0.0, 0.002, top_fraction=0.2)
    shuffled = Scenario(WINDOW_TEST, SIGNAL_SHUFFLED, 0.0, 0.002, top_fraction=0.2, seed=0)

    real = predictions_for(model, test, validate, frame)
    fake = predictions_for(shuffled, test, validate, frame)

    assert (fake > 0).sum() == (real > 0).sum()
    assert list(fake) != list(real)
    # deterministic per seed, different across seeds
    again = predictions_for(shuffled, test, validate, frame)
    other = predictions_for(Scenario(WINDOW_TEST, SIGNAL_SHUFFLED, 0.0, 0.002, top_fraction=0.2, seed=1), test, validate, frame)
    assert list(fake) == list(again) and list(fake) != list(other)


def test_sweep_command_adds_shuffled_rows_and_prediction_diagnostics(tmp_path: Path):
    swaps, model = _train(tmp_path)
    out_dir = tmp_path / "sweeps"

    result = CliRunner().invoke(
        cli,
        ["sweep", "--swaps", str(swaps), "--model", str(model), "--out-dir", str(out_dir),
         "--min-edges", "0", "--top-fractions", "0.1,0.5,1.0", "--shuffle-seeds", "0,1"],
    )

    assert result.exit_code == 0, result.output
    (saved,) = list(out_dir.glob("*.json"))
    record = json.loads(saved.read_text())
    shuffled = [r for r in record["scenarios"] if r["signal"] == SIGNAL_SHUFFLED]
    assert len(shuffled) == 2 * 2 * 2  # 2 windows x 2 seeds x (0.1 and 0.5)
    assert record["shuffle_seeds"] == [0, 1]
    for window in (WINDOW_VALIDATE, WINDOW_TEST):
        assert {"gross_correlation", "cost_correlation", "gross_std", "cost_std"} <= set(record["prediction_diagnostics"][window])
    assert record["label_horizon_seconds"] == 1800
