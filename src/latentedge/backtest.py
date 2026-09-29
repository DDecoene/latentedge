from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from latentedge import config
from latentedge.executor import simulate_fill
from latentedge.labeling import sort_swaps
from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result, roll_to_day
from latentedge.signal_client import SignalClient
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price


class BacktestInputs(NamedTuple):
    features: np.ndarray
    entry_prices: np.ndarray
    exit_prices: np.ndarray
    entry_swaps: list[dict[str, Any]]
    exit_swaps: list[dict[str, Any]]
    timestamps: np.ndarray


def _swap_dict(swaps: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = swaps.iloc[idx]
    return {
        "liquidity": int(row["liquidity"]),
        "sqrt_price_x96": int(row["sqrt_price_x96"]),
        "base_fee_wei": int(row["base_fee_wei"]),
    }


def build_backtest_inputs(rows: pd.DataFrame, swaps: pd.DataFrame, feature_columns: list[str]) -> BacktestInputs:
    """Turns assembled (labeled) bars into the arrays run_backtest replays.

    Fills come from the swap row positions the label itself recorded
    (entry_swap_idx / exit_swap_idx), looked up in the canonical swap
    ordering, so the backtest trades exactly the fills the label was
    computed from — never a second, possibly different, fill search.
    Features stay raw: the signal client standardizes them itself.
    """
    rows = rows.sort_values("bar_start", kind="stable")
    ordered = sort_swaps(swaps)
    prices = ordered["sqrt_price_x96"].map(lambda v: sqrt_price_x96_to_weth_usdc_price(int(v))).to_numpy(dtype="float64")

    entry_idx = rows["entry_swap_idx"].to_numpy(dtype="int64")
    exit_idx = rows["exit_swap_idx"].to_numpy(dtype="int64")

    return BacktestInputs(
        features=rows[feature_columns].to_numpy(dtype="float64"),
        entry_prices=prices[entry_idx],
        exit_prices=prices[exit_idx],
        entry_swaps=[_swap_dict(ordered, int(i)) for i in entry_idx],
        exit_swaps=[_swap_dict(ordered, int(i)) for i in exit_idx],
        timestamps=rows["bar_start"].to_numpy(),
    )


def daily_sharpe(equity_curve: np.ndarray, timestamps: np.ndarray) -> float:
    """Annualized Sharpe-like ratio (risk-free rate 0) of end-of-day equity
    returns. The equity curve has one point per bar, so it is first
    collapsed to the last value of each calendar day."""
    days = timestamps // 86_400
    last_of_day = np.flatnonzero(np.append(days[1:] != days[:-1], True))
    daily_equity = equity_curve[last_of_day]
    if len(daily_equity) < 3:
        return 0.0
    returns = daily_equity[1:] / daily_equity[:-1] - 1.0
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(365))


@dataclass(frozen=True)
class BacktestProgress:
    """A point-in-time view of a running backtest, for a live display.
    `events` are the trades settled and lockouts hit since the previous
    snapshot."""

    done: int
    total: int
    timestamp: int
    equity_usd: float
    peak_equity_usd: float
    max_drawdown_usd: float
    num_trades: int
    wins: int
    open_positions: int
    committed_usd: float
    lockout_days: int
    events: list[str] = field(default_factory=list)


def _noop_stage(*_: object) -> None:
    pass


def _noop_predictions(_: np.ndarray) -> None:
    pass


def _noop_progress(_: BacktestProgress) -> None:
    pass


@dataclass
class BacktestObserver:
    """Everything a caller can watch of a backtest run. Calling the observer
    itself reports a preparation stage (stage, done, total), so it can be
    passed wherever a plain progress callback is expected. Any callback may
    raise to abort the run."""

    on_stage: Callable[..., None] = _noop_stage
    on_predictions: Callable[[np.ndarray], None] = _noop_predictions
    on_progress: Callable[[BacktestProgress], None] = _noop_progress

    def __call__(self, *args: Any) -> None:
        self.on_stage(*args)


# Roughly this many progress snapshots per run, however long the window.
PROGRESS_SNAPSHOTS = 300


def _format_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%b %d %H:%M")


class BacktestResult(BaseModel):
    total_return_usd: float
    max_drawdown_usd: float
    win_rate: float
    num_trades: int
    equity_curve: list[float]


class _OpenPosition(BaseModel):
    exit_timestamp: int
    pnl_usd: float
    size_usd: float


def run_backtest(
    features: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    entry_swaps: list[dict[str, Any]],
    exit_swaps: list[dict[str, Any]],
    signal_client: SignalClient,
    guard: SafetyGuard,
    initial_equity_usd: float,
    timestamps: np.ndarray,
    observer: BacktestObserver | None = None,
) -> BacktestResult:
    """Backtest with realistic accounting: a position's P&L is realized at
    its actual exit time (entry + the label horizon), not the moment it
    opens, and new positions are sized against capital not already
    committed to still-open ones — a trade never sizes against equity the
    account doesn't actually have free. Inputs are assumed sorted by
    timestamp, matching how bars are produced upstream.
    """
    observer = observer or BacktestObserver()
    predictions = signal_client.predict_batch(features)
    observer.on_predictions(predictions)

    state = GuardState(equity_usd=initial_equity_usd, daily_loss_usd=0.0, current_day=0, locked_out=False)

    open_positions: list[_OpenPosition] = []
    committed_capital = 0.0

    equity_curve: list[float] = []
    wins = 0
    num_trades = 0
    peak_equity = initial_equity_usd
    max_drawdown = 0.0
    lockout_days = 0
    events: list[str] = []
    total_bars = len(features)
    snapshot_every = max(total_bars // PROGRESS_SNAPSHOTS, 1)

    def settle(position: "_OpenPosition") -> None:
        nonlocal state, committed_capital, wins, lockout_days
        state = roll_to_day(state, position.exit_timestamp)
        was_locked_out = state.locked_out
        state = record_trade_result(state, pnl_usd=position.pnl_usd, daily_loss_limit_fraction=guard.daily_loss_limit_fraction)
        committed_capital -= position.size_usd
        if position.pnl_usd > 0:
            wins += 1
        events.append(
            f"{_format_time(position.exit_timestamp)} closed ${position.size_usd:,.0f} position: "
            f"{position.pnl_usd:+,.2f} USD, equity ${state.equity_usd:,.2f}"
        )
        if state.locked_out and not was_locked_out:
            lockout_days += 1
            events.append(
                f"{_format_time(position.exit_timestamp)} daily loss limit hit "
                f"(${state.daily_loss_usd:,.2f}) — no new trades until the next day"
            )

    def report(done: int, timestamp: int) -> None:
        observer.on_progress(BacktestProgress(
            done=done, total=total_bars, timestamp=timestamp, equity_usd=state.equity_usd,
            peak_equity_usd=peak_equity, max_drawdown_usd=max_drawdown, num_trades=num_trades, wins=wins,
            open_positions=len(open_positions), committed_usd=committed_capital, lockout_days=lockout_days,
            events=list(events),
        ))
        events.clear()

    for i in range(len(features)):
        entry_timestamp = int(timestamps[i])

        # Settle any position whose exit time has arrived before this
        # new entry — a trade must not use capital, or influence the
        # daily-loss lockout, before it has actually closed.
        still_open: list[_OpenPosition] = []
        for position in open_positions:
            if position.exit_timestamp <= entry_timestamp:
                settle(position)
            else:
                still_open.append(position)
        open_positions = still_open

        size_usd, state = guard.size_position(state, predicted_return=float(predictions[i]), timestamp=entry_timestamp)

        # Cap by capital actually free — already-open positions have
        # claimed some of total equity, and a new one can't exceed what's
        # left, regardless of what the guard's fraction-of-equity
        # calculation alone would suggest.
        available_capital = max(state.equity_usd - committed_capital, 0.0)
        size_usd = min(size_usd, available_capital)

        if size_usd > 0:
            pnl = simulate_fill(size_usd, float(entry_prices[i]), float(exit_prices[i]), entry_swaps[i], exit_swaps[i])
            exit_timestamp = entry_timestamp + config.LABEL_HORIZON_SECONDS
            open_positions.append(_OpenPosition(exit_timestamp=exit_timestamp, pnl_usd=pnl, size_usd=size_usd))
            committed_capital += size_usd
            num_trades += 1

        equity_curve.append(state.equity_usd)
        peak_equity = max(peak_equity, state.equity_usd)
        max_drawdown = max(max_drawdown, peak_equity - state.equity_usd)

        if (i + 1) % snapshot_every == 0:
            report(i + 1, entry_timestamp)

    # Settle whatever is still open at the end of the backtest window —
    # every opened trade must be accounted for in the final result.
    for position in open_positions:
        settle(position)
    if total_bars:
        report(total_bars, int(timestamps[-1]))

    total_return_usd = state.equity_usd - initial_equity_usd
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    return BacktestResult(
        total_return_usd=total_return_usd,
        max_drawdown_usd=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity_curve,
    )
