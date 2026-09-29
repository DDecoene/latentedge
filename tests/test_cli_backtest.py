import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from latentedge.cli import cli
from latentedge.ingest.progress import write_progress
from latentedge.training_data import FEATURE_COLUMNS
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


def test_backtest_after_train_runs_the_backtest_once_training_finishes(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)

    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5", "--backtest-after-train"]
    )

    assert result.exit_code == 0, result.output
    assert "total return" in result.output.lower()
    assert Path(str(model) + ".backtest.json").exists()


def test_train_without_the_flag_does_not_backtest(tmp_path: Path):
    _, model = _train(tmp_path)
    assert not Path(str(model) + ".backtest.json").exists()


def test_train_records_gross_correlation_and_the_default_setup(tmp_path: Path):
    _, model = _train(tmp_path)
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())

    assert {"gross_correlation", "cost_correlation"} <= set(metrics["splits"]["test"])
    assert metrics["label_horizon_seconds"] == 1800
    assert metrics["feature_columns"] == FEATURE_COLUMNS


def test_excluded_features_are_left_out_of_training_and_the_backtest_follows_the_model(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)
    runner = CliRunner()

    trained = runner.invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5"],
        env={"LATENTEDGE_TRAIN_EXCLUDE_FEATURES": "base_fee_gwei, volatility"},
    )
    assert trained.exit_code == 0, trained.output
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())
    assert metrics["feature_columns"] == [
        c for c in FEATURE_COLUMNS if c not in ("base_fee_gwei", "volatility")
    ]

    # no exclusion in the environment now: the backtest must still use the model's five features
    replayed = runner.invoke(cli, ["backtest", "--swaps", str(swaps), "--model", str(model)])
    assert replayed.exit_code == 0, replayed.output


def test_unknown_or_total_feature_exclusion_is_refused_before_any_work(tmp_path: Path):
    runner = CliRunner()
    args = ["train", "--swaps", str(tmp_path / "s.parquet"), "--out", str(tmp_path / "m.safetensors")]

    unknown = runner.invoke(cli, args, env={"LATENTEDGE_TRAIN_EXCLUDE_FEATURES": "gas"})
    everything = runner.invoke(
        cli, args, env={"LATENTEDGE_TRAIN_EXCLUDE_FEATURES": ",".join(FEATURE_COLUMNS)}
    )

    assert unknown.exit_code != 0 and "unknown feature" in unknown.output
    assert everything.exit_code != 0 and "no features" in everything.output


def test_a_longer_label_horizon_is_recorded_and_reused_by_the_backtest_and_sweep(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps, n_minutes=9000)
    runner = CliRunner()

    trained = runner.invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5"],
        env={"LATENTEDGE_LABEL_HORIZON_MINUTES": "120", "LATENTEDGE_LABEL_BARRIER_STDS": "5"},
    )
    assert trained.exit_code == 0, trained.output
    metrics = json.loads(Path(str(model) + ".metrics.json").read_text())
    assert metrics["label_horizon_seconds"] == 7200 and metrics["label_barrier_stds"] == 5.0

    # the environment is back to defaults: the model's own recorded settings must win
    replayed = runner.invoke(cli, ["backtest", "--swaps", str(swaps), "--model", str(model)])
    assert replayed.exit_code == 0, replayed.output
    assert json.loads(Path(str(model) + ".backtest.json").read_text())["bars"] == metrics["splits"]["test"]["n"]

    swept = runner.invoke(
        cli, ["sweep", "--swaps", str(swaps), "--model", str(model), "--out-dir", str(tmp_path / "sw"),
              "--min-edges", "0", "--top-fractions", "0.5", "--shuffle-seeds", ""],
    )
    assert swept.exit_code == 0, swept.output
    (saved,) = list((tmp_path / "sw").glob("*.json"))
    assert json.loads(saved.read_text())["label_horizon_seconds"] == 7200


@pytest.mark.parametrize("env", [
    {"LATENTEDGE_LABEL_HORIZON_MINUTES": "0"},
    {"LATENTEDGE_LABEL_HORIZON_MINUTES": "soon"},
    {"LATENTEDGE_LABEL_BARRIER_STDS": "-1"},
])
def test_bad_label_settings_are_refused_before_any_work(tmp_path: Path, env: dict):
    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(tmp_path / "s.parquet"), "--out", str(tmp_path / "m.safetensors")], env=env
    )
    assert result.exit_code != 0
    assert "LATENTEDGE_LABEL" in result.output


def test_train_can_chain_the_sweep_and_then_the_backtest(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)

    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5",
              "--sweep-after-train", "--backtest-after-train"],
        env={"LATENTEDGE_SWEEP_OUT_DIR": str(tmp_path / "sweeps"), "LATENTEDGE_SWEEP_SHUFFLE_SEEDS": "0"},
    )

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "sweeps").glob("*.json"))) == 1
    assert Path(str(model) + ".backtest.json").exists()
    assert result.output.index("scenarios saved") < result.output.index("backtest over")


def _write_trending_swaps(path: Path, n_minutes: int = 8000, seed: int = 5) -> None:
    """Swaps whose price follows a hidden drift that flips sign every couple
    of hours, so the last few bars' return genuinely predicts the next
    half hour's. A pipeline that cannot find this has a bug."""
    rng = np.random.default_rng(seed)
    rows = []
    price, drift, log_index = 3000.0, 0.0, 0
    for minute in range(n_minutes):
        if minute % 120 == 0:
            drift = rng.choice([-1.0, 1.0]) * 0.0004
        for _ in range(2):
            price *= 1 + drift / 2 + rng.normal(0, 0.0004)
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


def test_the_pipeline_finds_a_planted_direction_signal(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_trending_swaps(swaps)

    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "500", "--no-sweep-after-train"]
    )

    assert result.exit_code == 0, result.output
    splits = json.loads(Path(str(model) + ".metrics.json").read_text())["splits"]
    for name in ("validate", "test"):
        assert splits[name]["gross_correlation"] > 0.15, (name, splits[name])


def test_feature_groups_can_be_excluded_by_name(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)

    trained = CliRunner().invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "5", "--no-sweep-after-train"],
        env={"LATENTEDGE_TRAIN_EXCLUDE_FEATURES": "order_flow"},
    )

    assert trained.exit_code == 0, trained.output
    columns = json.loads(Path(str(model) + ".metrics.json").read_text())["feature_columns"]
    assert columns == ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap", "base_fee_gwei"]


def _write_flow_led_swaps(path: Path, n_minutes: int = 8000, seed: int = 9) -> None:
    """Swaps where the price is a random walk with heavy noise, but who is
    buying leads it: a regime that flips every two hours tilts both the
    direction of the trades (buys vs sells) and, slightly, the price. The
    price alone is too noisy to show the regime; the order flow shows it
    plainly. Only a model that reads the order flow can find it."""
    rng = np.random.default_rng(seed)
    rows = []
    price, regime, log_index = 3000.0, 0.0, 0
    for minute in range(n_minutes):
        if minute % 120 == 0:
            regime = rng.choice([-1.0, 1.0])
        for _ in range(4):
            price *= 1 + regime * 0.0001 + rng.normal(0, 0.0015)
            buy = rng.random() < 0.5 + 0.4 * regime
            size = float(abs(rng.normal(5000, 1000)) * 10**6)
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
                    "amount0": size if buy else -size,
                    "amount1": 0.0,
                    "base_fee_wei": 15_000_000_000,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    write_progress(path, [(int(df["block_number"].min()), int(df["block_number"].max()))])


def test_order_flow_finds_a_signal_that_price_alone_cannot(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    _write_flow_led_swaps(swaps)

    def gross_correlations(exclude: str) -> dict[str, float]:
        model = tmp_path / f"model-{exclude}.safetensors"
        result = CliRunner().invoke(
            cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "500", "--no-sweep-after-train"],
            env={"LATENTEDGE_TRAIN_EXCLUDE_FEATURES": exclude},
        )
        assert result.exit_code == 0, result.output
        splits = json.loads(Path(str(model) + ".metrics.json").read_text())["splits"]
        return {name: splits[name]["gross_correlation"] for name in ("validate", "test")}

    with_flow = gross_correlations("returns")  # order flow and the rest, no lagged returns
    price_only = gross_correlations("order_flow")

    for name in ("validate", "test"):
        assert with_flow[name] > 0.12, (name, with_flow)
        assert with_flow[name] > price_only[name] + 0.05, (name, with_flow, price_only)


def _write_costly_gas_swaps(path: Path, n_minutes: int = 8000, seed: int = 21) -> None:
    """A pure random walk (no direction to find) whose gas price swings a lot
    from one half hour to the next, so what a trade costs is very predictable."""
    rng = np.random.default_rng(seed)
    rows = []
    price, gas_gwei, log_index = 3000.0, 20.0, 0
    for minute in range(n_minutes):
        if minute % 30 == 0:
            gas_gwei = float(rng.uniform(5, 200))
        price *= 1 + rng.normal(0, 0.0012)
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
                "base_fee_wei": int(gas_gwei * 1_000_000_000),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    write_progress(path, [(int(df["block_number"].min()), int(df["block_number"].max()))])


def test_a_net_trained_model_learns_cost_and_a_gross_trained_one_does_not(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    _write_costly_gas_swaps(swaps)

    def train_on(target: str) -> dict:
        model = tmp_path / f"model-{target}.safetensors"
        result = CliRunner().invoke(
            cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "500", "--no-sweep-after-train"],
            env={"LATENTEDGE_TRAIN_TARGET": target},
        )
        assert result.exit_code == 0, result.output
        return json.loads(Path(str(model) + ".metrics.json").read_text())

    net = train_on("net")
    gross = train_on("gross")

    assert net["target"] == "net" and gross["target"] == "gross"
    # Gas changes only every half hour here, so a window holds few independent
    # points and a lone correlation is noisy; compare the two models instead.
    net_cost = net["splits"]["test"]["cost_correlation"]
    gross_cost = gross["splits"]["test"]["cost_correlation"]
    assert net_cost < -0.6  # it learned the cost
    assert abs(gross_cost) < abs(net_cost) - 0.25  # far less than the model that was asked to
    # a gross-trained model's own correlation is with the price move
    assert gross["splits"]["test"]["correlation"] == pytest.approx(gross["splits"]["test"]["gross_correlation"], abs=1e-5)


def test_gross_is_the_default_training_target(tmp_path: Path):
    _, model = _train(tmp_path)
    assert json.loads(Path(str(model) + ".metrics.json").read_text())["target"] == "gross"


def test_a_bad_training_target_is_refused_before_any_work(tmp_path: Path):
    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(tmp_path / "s.parquet"), "--out", str(tmp_path / "m.safetensors")],
        env={"LATENTEDGE_TRAIN_TARGET": "profit"},
    )
    assert result.exit_code != 0 and "LATENTEDGE_TRAIN_TARGET" in result.output


def test_the_backtest_and_sweep_return_real_units_for_a_gross_trained_model(tmp_path: Path):
    swaps, model = _train(tmp_path)  # gross by default
    swept = CliRunner().invoke(
        cli, ["sweep", "--swaps", str(swaps), "--model", str(model), "--out-dir", str(tmp_path / "sw"),
              "--min-edges", "0", "--top-fractions", "0.5", "--shuffle-seeds", ""],
    )
    assert swept.exit_code == 0, swept.output


def test_the_network_size_is_configurable_recorded_and_reused_by_the_sweep(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    model = tmp_path / "model.safetensors"
    _write_synthetic_swaps(swaps)
    runner = CliRunner()

    trained = runner.invoke(
        cli, ["train", "--swaps", str(swaps), "--out", str(model), "--epochs", "20", "--no-sweep-after-train"],
        env={"LATENTEDGE_TRAIN_HIDDEN": "32,16"},
    )
    assert trained.exit_code == 0, trained.output
    assert json.loads(Path(str(model) + ".metrics.json").read_text())["hidden_sizes"] == [32, 16]

    swept = runner.invoke(
        cli, ["sweep", "--swaps", str(swaps), "--model", str(model), "--out-dir", str(tmp_path / "sw"),
              "--min-edges", "0", "--top-fractions", "0.5", "--shuffle-seeds", ""],
    )
    assert swept.exit_code == 0, swept.output


@pytest.mark.parametrize("value", ["0", "abc", "8,8,8,8,8", "-4", ""])
def test_a_bad_network_size_is_refused_before_any_work(tmp_path: Path, value: str):
    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(tmp_path / "s.parquet"), "--out", str(tmp_path / "m.safetensors")],
        env={"LATENTEDGE_TRAIN_HIDDEN": value},
    )
    assert result.exit_code != 0 and "LATENTEDGE_TRAIN_HIDDEN" in result.output
