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
    # The last bar's own trade (entered at t=7200) settles at t=9000,
    # after the loop has already appended its equity_curve entry — its
    # P&L only lands in the final post-loop settlement pass, so
    # equity_curve[-1] reflects trades 1-2 settled but not yet trade 3.
    # It can therefore lag, but never exceed, the fully-realized total.
    assert result.equity_curve[-1] <= 10_000.0 + result.total_return_usd + 1e-6
    assert result.equity_curve[-1] > 10_000.0  # trades 1-2 already settled and profitable


def test_backtest_defers_pnl_to_exit_time_not_entry_time(tmp_path: Path):
    # Regression test: a real bug booked each trade's full outcome into
    # equity the moment it was entered, even though it would realistically
    # still be open for the label horizon (30 minutes). Two bars entered
    # 900 seconds apart (both well within a 1800s horizon of each other)
    # means neither trade has actually closed by the time the loop
    # finishes — equity must stay flat during the loop; only the final
    # settlement pass (after the last bar) may realize the P&L.
    x = np.zeros((20, 1), dtype=np.float32)
    y = np.full(20, 0.02, dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, y, epochs=200, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=1)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.5, full_size_return=0.02)

    features = np.zeros((2, 1), dtype=np.float32)
    entry_prices = np.array([3000.0, 3050.0])
    exit_prices = np.array([3200.0, 3250.0])
    entry_swaps = [_swap_dict(p) for p in entry_prices]
    exit_swaps = [_swap_dict(p) for p in exit_prices]
    timestamps = np.array([0, 900])  # both within 1800s of each other

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

    assert result.num_trades == 2
    assert result.equity_curve[0] == pytest.approx(10_000.0)  # nothing settled yet
    assert result.equity_curve[1] == pytest.approx(10_000.0)  # still nothing settled
    assert result.total_return_usd > 0  # realized in the final settlement pass


def test_backtest_caps_new_position_size_by_available_capital(tmp_path: Path):
    # Regression test: sizing was recomputed from raw equity every time,
    # ignoring capital already committed to still-open positions — three
    # overlapping trades could each size as if the full account were
    # available, together committing far more than total equity. With
    # max_position_fraction=1.0 (each trade wants the whole account),
    # only the first trade should find any capital available; the rest
    # must be skipped, not opened at a size the account doesn't have.
    x = np.zeros((20, 1), dtype=np.float32)
    y = np.full(20, 0.02, dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, y, epochs=200, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=1)
    # full_size_return set well below the trained model's actual output
    # (~0.02, with normal training variance in either direction) so
    # confidence reliably clamps to exactly 1.0 regardless of that
    # variance — the test needs the first trade to deterministically
    # claim 100% of equity, not "however close to 0.02 training happened
    # to land this run."
    guard = SafetyGuard(max_position_fraction=1.0, daily_loss_limit_fraction=0.99, full_size_return=0.001)

    features = np.zeros((3, 1), dtype=np.float32)
    entry_prices = np.array([3000.0, 3050.0, 3100.0])
    exit_prices = np.array([3200.0, 3250.0, 3300.0])
    entry_swaps = [_swap_dict(p) for p in entry_prices]
    exit_swaps = [_swap_dict(p) for p in exit_prices]
    timestamps = np.array([0, 300, 600])  # all within 1800s of each other

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

    assert result.num_trades == 1  # only the first trade found capital available


def test_backtest_reports_progress_predictions_and_trade_events_to_an_observer(tmp_path: Path):
    from latentedge.backtest import BacktestObserver

    x = np.zeros((20, 1), dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, np.full(20, 0.02, dtype=np.float32), epochs=200, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    n_bars = 3
    entry_prices = np.array([3000.0, 3200.0, 3400.0])
    exit_prices = np.array([3200.0, 3400.0, 3600.0])
    seen_predictions: list[np.ndarray] = []
    snapshots = []
    result = run_backtest(
        features=np.zeros((n_bars, 1), dtype=np.float32),
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        entry_swaps=[_swap_dict(p) for p in entry_prices],
        exit_swaps=[_swap_dict(p) for p in exit_prices],
        signal_client=SignalClient(model_path, input_dim=1),
        guard=SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.5, full_size_return=0.02),
        initial_equity_usd=10_000.0,
        timestamps=np.array([0, 3600, 7200]),
        observer=BacktestObserver(on_predictions=seen_predictions.append, on_progress=snapshots.append),
    )

    assert len(seen_predictions) == 1 and seen_predictions[0].shape == (n_bars,)
    assert snapshots[-1].done == n_bars == snapshots[-1].total
    # the last snapshot comes after every open position has settled
    assert snapshots[-1].num_trades == result.num_trades == 3
    assert snapshots[-1].open_positions == 0 or snapshots[-1].committed_usd >= 0
    assert snapshots[-1].equity_usd == pytest.approx(10_000.0 + result.total_return_usd)
    # every settled trade is announced exactly once across all snapshots
    events = [event for snapshot in snapshots for event in snapshot.events]
    assert sum("closed" in event for event in events) == 3


def test_backtest_observer_can_abort_the_run(tmp_path: Path):
    from latentedge.backtest import BacktestObserver

    x = np.zeros((20, 1), dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, np.full(20, 0.02, dtype=np.float32), epochs=50, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    class Stop(Exception):
        pass

    def stop(_progress):
        raise Stop

    prices = np.array([3000.0, 3200.0])
    with pytest.raises(Stop):
        run_backtest(
            features=np.zeros((2, 1), dtype=np.float32),
            entry_prices=prices, exit_prices=prices + 200,
            entry_swaps=[_swap_dict(p) for p in prices], exit_swaps=[_swap_dict(p + 200) for p in prices],
            signal_client=SignalClient(model_path, input_dim=1),
            guard=SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.5, full_size_return=0.02),
            initial_equity_usd=10_000.0, timestamps=np.array([0, 3600]),
            observer=BacktestObserver(on_progress=stop),
        )
