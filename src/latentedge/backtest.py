from typing import Any

import numpy as np
from pydantic import BaseModel

from latentedge.executor import simulate_fill
from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result
from latentedge.signal_client import SignalClient


class BacktestResult(BaseModel):
    total_return_usd: float
    max_drawdown_usd: float
    win_rate: float
    num_trades: int
    equity_curve: list[float]


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
    predictions = signal_client.predict_batch(features)

    state = GuardState(equity_usd=initial_equity_usd, daily_loss_usd=0.0, current_day=0, locked_out=False)

    equity_curve: list[float] = []
    wins = 0
    num_trades = 0
    peak_equity = initial_equity_usd
    max_drawdown = 0.0

    for i in range(len(features)):
        size_usd, state = guard.size_position(state, predicted_return=float(predictions[i]), timestamp=int(timestamps[i]))

        if size_usd > 0:
            pnl = simulate_fill(size_usd, float(entry_prices[i]), float(exit_prices[i]), entry_swaps[i], exit_swaps[i])
            num_trades += 1
            if pnl > 0:
                wins += 1
            state = record_trade_result(state, pnl_usd=pnl, daily_loss_limit_fraction=guard.daily_loss_limit_fraction)

        equity_curve.append(state.equity_usd)
        peak_equity = max(peak_equity, state.equity_usd)
        max_drawdown = max(max_drawdown, peak_equity - state.equity_usd)

    total_return_usd = state.equity_usd - initial_equity_usd
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    return BacktestResult(
        total_return_usd=total_return_usd,
        max_drawdown_usd=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity_curve,
    )
