from typing import Any

import numpy as np
from pydantic import BaseModel

from latentedge import config
from latentedge.executor import simulate_fill
from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result, roll_to_day
from latentedge.signal_client import SignalClient


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
) -> BacktestResult:
    """Backtest with realistic accounting: a position's P&L is realized at
    its actual exit time (entry + the label horizon), not the moment it
    opens, and new positions are sized against capital not already
    committed to still-open ones — a trade never sizes against equity the
    account doesn't actually have free. Inputs are assumed sorted by
    timestamp, matching how bars are produced upstream.
    """
    predictions = signal_client.predict_batch(features)

    state = GuardState(equity_usd=initial_equity_usd, daily_loss_usd=0.0, current_day=0, locked_out=False)

    open_positions: list[_OpenPosition] = []
    committed_capital = 0.0

    equity_curve: list[float] = []
    wins = 0
    num_trades = 0
    peak_equity = initial_equity_usd
    max_drawdown = 0.0

    def settle(position: "_OpenPosition") -> None:
        nonlocal state, committed_capital, wins
        state = roll_to_day(state, position.exit_timestamp)
        state = record_trade_result(state, pnl_usd=position.pnl_usd, daily_loss_limit_fraction=guard.daily_loss_limit_fraction)
        committed_capital -= position.size_usd
        if position.pnl_usd > 0:
            wins += 1

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

    # Settle whatever is still open at the end of the backtest window —
    # every opened trade must be accounted for in the final result.
    for position in open_positions:
        settle(position)

    total_return_usd = state.equity_usd - initial_equity_usd
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    return BacktestResult(
        total_return_usd=total_return_usd,
        max_drawdown_usd=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity_curve,
    )
