"""Scenario sweeps over the backtest: the same model replayed under a grid of
trading rules, on both the validation and the untouched test window, next to
reference strategies — kept on disk so runs can be compared later."""

import hashlib
import itertools
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentedge.backtest import BacktestInputs, daily_sharpe, run_backtest
from latentedge.safety_guard import SafetyGuard

# Signals a scenario can trade on: the trained model, a strategy that takes
# every bar at full size (what trading with no model costs), and an oracle
# that "predicts" each bar's realized net return (the ceiling on what any
# model could earn under the same costs and sizing).
SIGNAL_MODEL = "model"
SIGNAL_ALWAYS = "always"
SIGNAL_ORACLE = "oracle"

WINDOW_VALIDATE = "validate"
WINDOW_TEST = "test"


@dataclass(frozen=True)
class Scenario:
    window: str
    signal: str
    min_edge: float
    full_size_return: float
    # For the model only: trade just this share of bars, the ones it ranks
    # highest, at full size — whatever the predictions' absolute level.
    # A model that never predicts a positive return still ranks bars, and
    # this asks whether the ranking is worth anything. None = no rank rule.
    top_fraction: float | None = None


@dataclass
class SweepObserver:
    """What a caller can watch of a sweep. Calling the observer reports a
    preparation stage or the scenario about to run (label, done, total);
    on_scenario receives each finished scenario's result row. Either may
    raise to abort."""

    on_stage: Callable[..., None] = lambda *_: None
    on_scenario: Callable[[dict[str, Any]], None] = lambda _row: None

    def __call__(self, *args: Any) -> None:
        self.on_stage(*args)


class FixedPredictions:
    """A signal client that returns predictions computed up front, so a grid
    of scenarios pays for the model's forward pass once per window."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        return self._predictions


def build_scenarios(
    min_edges: list[float], full_size_returns: list[float], top_fractions: list[float]
) -> list[Scenario]:
    """The model under each absolute rule (a minimum predicted return, at each
    sizing) and each rank rule (its top share of bars), then the two reference
    strategies at the first sizing, each on both windows (validate first: it
    is where a rule is chosen, the test window is only for reporting the
    choice)."""
    scenarios: list[Scenario] = []
    for window in (WINDOW_VALIDATE, WINDOW_TEST):
        for min_edge, full_size in itertools.product(min_edges, full_size_returns):
            scenarios.append(Scenario(window, SIGNAL_MODEL, min_edge, full_size))
        for fraction in top_fractions:
            scenarios.append(Scenario(window, SIGNAL_MODEL, 0.0, full_size_returns[0], top_fraction=fraction))
        scenarios.append(Scenario(window, SIGNAL_ALWAYS, 0.0, full_size_returns[0]))
        scenarios.append(Scenario(window, SIGNAL_ORACLE, 0.0, full_size_returns[0]))
    return scenarios


def split_windows(assembled: pd.DataFrame, test_start: int, validate_bars: int) -> dict[str, pd.DataFrame]:
    """The two evaluation windows, in the model's own terms: the test window
    starts where training recorded it did, and the validation window is the
    validate_bars bars just before it — the same bars the model was validated
    on, however much has been ingested since."""
    ordered = assembled.sort_values("bar_start", kind="stable")
    before_test = ordered[ordered["bar_start"] < test_start]
    return {
        WINDOW_VALIDATE: before_test.iloc[max(len(before_test) - validate_bars, 0):],
        WINDOW_TEST: ordered[ordered["bar_start"] >= test_start],
    }


def predictions_for(
    scenario: Scenario, model_predictions: np.ndarray, validate_predictions: np.ndarray, window: pd.DataFrame
) -> np.ndarray:
    """What the guard sees for each bar of the scenario's window. A rank rule
    turns the model's predictions into full-size buys on the bars above the
    cutoff and nothing elsewhere; the cutoff comes from the validation
    window's predictions so it is fixed before the test window is looked at."""
    if scenario.signal == SIGNAL_MODEL:
        if scenario.top_fraction is None:
            return model_predictions
        if scenario.top_fraction >= 1.0:
            return np.full(len(model_predictions), scenario.full_size_return, dtype="float64")
        cutoff = float(np.quantile(validate_predictions, 1.0 - scenario.top_fraction))
        chosen = model_predictions > cutoff
        return np.where(chosen, scenario.full_size_return, 0.0)
    if scenario.signal == SIGNAL_ALWAYS:
        return np.full(len(window), scenario.full_size_return, dtype="float64")
    if scenario.signal == SIGNAL_ORACLE:
        return window["net_return"].to_numpy(dtype="float64")
    raise ValueError(f"unknown signal {scenario.signal!r}")


def run_scenario(
    scenario: Scenario,
    inputs: BacktestInputs,
    predictions: np.ndarray,
    max_position_fraction: float,
    daily_loss_limit_fraction: float,
    initial_equity: float,
) -> dict[str, Any]:
    guard = SafetyGuard(
        max_position_fraction=max_position_fraction,
        daily_loss_limit_fraction=daily_loss_limit_fraction,
        full_size_return=scenario.full_size_return,
        min_edge=scenario.min_edge,
    )
    result = run_backtest(
        features=inputs.features,
        entry_prices=inputs.entry_prices,
        exit_prices=inputs.exit_prices,
        entry_swaps=inputs.entry_swaps,
        exit_swaps=inputs.exit_swaps,
        signal_client=FixedPredictions(predictions),  # type: ignore[arg-type]
        guard=guard,
        initial_equity_usd=initial_equity,
        timestamps=inputs.timestamps,
    )
    return {
        **asdict(scenario),
        "bars": len(inputs.timestamps),
        "window_start": int(inputs.timestamps[0]),
        "window_end": int(inputs.timestamps[-1]),
        "num_trades": result.num_trades,
        "total_return_usd": result.total_return_usd,
        "total_return_fraction": result.total_return_usd / initial_equity,
        "max_drawdown_usd": result.max_drawdown_usd,
        "win_rate": result.win_rate,
        "avg_pnl_usd": result.total_return_usd / result.num_trades if result.num_trades else 0.0,
        "sharpe": daily_sharpe(np.array(result.equity_curve), inputs.timestamps),
    }


def select_on_validate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The model scenario with the best validation return, and that same rule's
    result on the test window — the only honest out-of-sample number a sweep
    yields, since the rule was chosen without looking at the test window."""
    def key(row: dict[str, Any]) -> tuple[float, float, float | None]:
        return (row["min_edge"], row["full_size_return"], row.get("top_fraction"))

    validate = [r for r in rows if r["window"] == WINDOW_VALIDATE and r["signal"] == SIGNAL_MODEL]
    if not validate:
        return None
    best = max(validate, key=lambda r: r["total_return_usd"])
    test = next(
        (r for r in rows if r["window"] == WINDOW_TEST and r["signal"] == SIGNAL_MODEL and key(r) == key(best)), None
    )
    return (best, test) if test is not None else None


def _git_state() -> str | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}+dirty" if dirty else commit


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sweep(out_dir: Path, meta: dict[str, Any], rows: list[dict[str, Any]], now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> Path:
    """Saves one sweep as <out_dir>/<UTC timestamp>.json (metadata + every
    scenario row) and a .csv of the same rows. Never overwrites an earlier
    sweep — comparing runs is the point of keeping them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"{stamp}.json"
    counter = 1
    while json_path.exists():
        counter += 1
        json_path = out_dir / f"{stamp}-{counter}.json"
    document = {"created_at": now().isoformat(), "git": _git_state(), **meta, "scenarios": rows}
    json_path.write_text(json.dumps(document, indent=2))
    pd.DataFrame(rows).to_csv(json_path.with_suffix(".csv"), index=False)
    return json_path


def describe_rule(row: dict[str, Any]) -> str:
    if row.get("top_fraction") is not None:
        return f"top {row['top_fraction']:.0%} of bars"
    return f"min edge {row['min_edge']:.2%}, full size {row['full_size_return']:.2%}"


def describe_sweep(result: dict) -> str:
    selection = result["selection"]
    if selection is None:
        return f"{len(result['scenarios'])} scenarios saved to {result['path']}"
    best, test = selection
    return (
        f"{len(result['scenarios'])} scenarios saved to {result['path']}. Best on validate: "
        f"{describe_rule(best)} ({best['total_return_fraction']:+.2%}) "
        f"-> {test['total_return_fraction']:+.2%} on test ({test['num_trades']:,} trades)"
    )
