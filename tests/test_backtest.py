from pathlib import Path

import numpy as np
import pytest

from latentedge.backtest import run_backtest
from latentedge.model import NetReturnRegressor, save, train
from latentedge.safety_guard import SafetyGuard
from latentedge.signal_client import SignalClient
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap_dict(human_price: float) -> dict:
    # See Task 2's ruling: price_to_sqrt_price_x96 already decimal-adjusts
    # internally, so pass 1.0 / human_price directly, no extra scaling.
    raw_price = 1.0 / human_price
    return {"sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18), "liquidity": 10**18, "base_fee_wei": 20_000_000_000}


def test_backtest_produces_hand_computable_results(tmp_path: Path):
    # Train a trivial model that always predicts a positive number, by
    # fitting to an all-positive constant target — makes trade outcomes
    # (always taken) hand-computable from simulate_fill's own math.
    x = np.zeros((20, 1), dtype=np.float32)
    y = np.full(20, 0.02, dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, y, epochs=200, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=1)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.5, full_size_return=0.02)

    # Moves sized well above real-world cost drag (fees/slippage/gas at
    # $1000 position size) — a ~1.6% move looked "obviously profitable"
    # on paper but nets negative once ~20 gwei gas is accounted for
    # (verified directly against simulate_fill); a ~6-7% move, matching
    # what Task 15's executor tests already confirmed profitable, is
    # unambiguous.
    n_bars = 3
    features = np.zeros((n_bars, 1), dtype=np.float32)
    entry_prices = np.array([3000.0, 3200.0, 3400.0])
    exit_prices = np.array([3200.0, 3400.0, 3600.0])
    entry_swaps = [_swap_dict(p) for p in entry_prices]
    exit_swaps = [_swap_dict(p) for p in exit_prices]
    timestamps = np.array([0, 3600, 7200])

    result = run_backtest(
        features=features,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        entry_swaps=entry_swaps,
        exit_swaps=exit_swaps,
        signal_client=client,
        guard=guard,
        initial_equity_usd=10_000.0,
        timestamps=timestamps,
    )

    assert result.num_trades == 3
    assert result.total_return_usd > 0  # every simulated move here is profitable
    assert 0.0 <= result.win_rate <= 1.0
    assert len(result.equity_curve) == 3
    assert result.equity_curve[-1] == pytest.approx(10_000.0 + result.total_return_usd)
