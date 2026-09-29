import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from latentedge.cli import cli
from latentedge.ingest.progress import write_progress
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _write_synthetic_swaps(path: Path, n_minutes: int = 6000, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    price = 3000.0
    log_index = 0
    for minute in range(n_minutes):
        for _ in range(int(rng.integers(1, 4))):
            price *= 1 + rng.normal(0, 0.0007)
            log_index += 1
            rows.append(
                {
                    "block_number": 1_000 + minute // 12,
                    "timestamp": 1_700_000_000 + minute * 60,
                    "tx_hash": f"0x{log_index:064x}",
                    "log_index": log_index,
                    "sqrt_price_x96": str(price_to_sqrt_price_x96(1.0 / price, decimals0=6, decimals1=18)),
                    "tick": 0,
                    "liquidity": str(10**18),
                    "amount0": float(abs(rng.normal(5000, 1000)) * 10**6),
                    "amount1": 0.0,
                    "base_fee_wei": 15_000_000_000,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    write_progress(path, [(int(df["block_number"].min()), int(df["block_number"].max()))])


def _train(tmp_path: Path) -> tuple[Path, Path]:
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)
    result = CliRunner().invoke(cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5"])
    assert result.exit_code == 0, result.output
    return swaps, model


def test_train_records_where_the_untouched_test_window_starts(tmp_path: Path):
    _, model = _train(tmp_path)
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())
    assert isinstance(metrics["test_start"], int)
    assert metrics["test_start"] > 1_700_000_000


def test_backtest_replays_exactly_the_recorded_test_window_and_writes_a_summary(tmp_path: Path):
    swaps, model = _train(tmp_path)
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())

    result = CliRunner().invoke(cli, ["backtest", "--swaps", str(swaps), "--model", str(model)])

    assert result.exit_code == 0, result.output
    summary = json.loads(Path(str(model) + ".backtest.json").read_text())
    assert summary["bars"] == metrics["splits"]["test"]["n"]
    assert summary["test_start"] == metrics["test_start"]
    for key in ("total_return_usd", "max_drawdown_usd", "win_rate", "num_trades", "sharpe", "initial_equity_usd"):
        assert key in summary
    assert "total return" in result.output.lower()


def test_backtest_still_replays_the_original_test_window_after_more_data_is_ingested(tmp_path: Path):
    swaps, model = _train(tmp_path)
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())
    original_bars = metrics["splits"]["test"]["n"]

    # newer history arrives after training; a fresh 70/15/15 split would now
    # move the test boundary earlier, into data the model trained on
    _write_synthetic_swaps(swaps, n_minutes=9000)

    result = CliRunner().invoke(cli, ["backtest", "--swaps", str(swaps), "--model", str(model)])

    assert result.exit_code == 0, result.output
    summary = json.loads(Path(str(model) + ".backtest.json").read_text())
    assert summary["test_start"] == metrics["test_start"]
    assert summary["bars"] > original_bars  # the original window plus the newer bars, none earlier


def test_backtest_refuses_a_model_that_never_recorded_its_test_window(tmp_path: Path):
    swaps, model = _train(tmp_path)
    metrics_path = Path(str(model) + ".metrics.json")
    metrics = json.loads(metrics_path.read_text())
    del metrics["test_start"]
    metrics_path.write_text(json.dumps(metrics))

    result = CliRunner().invoke(cli, ["backtest", "--swaps", str(swaps), "--model", str(model)])

    assert result.exit_code != 0
    assert "test window" in result.output.lower()
    assert not Path(str(model) + ".backtest.json").exists()


def test_backtest_guard_parameters_come_from_env_vars(tmp_path: Path):
    swaps, model = _train(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["backtest", "--swaps", str(swaps), "--model", str(model)],
        env={"LATENTEDGE_BACKTEST_INITIAL_EQUITY": "5000", "LATENTEDGE_BACKTEST_MAX_POSITION_FRACTION": "0.05"},
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(Path(str(model) + ".backtest.json").read_text())
    assert summary["initial_equity_usd"] == 5000
    assert summary["max_position_fraction"] == 0.05
