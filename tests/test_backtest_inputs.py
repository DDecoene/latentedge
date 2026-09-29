import numpy as np
import pandas as pd

from latentedge import config
from latentedge.backtest import build_backtest_inputs
from latentedge.bars import build_bars
from latentedge.executor import simulate_fill
from latentedge.training_data import FEATURE_COLUMNS, assemble_training_data
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swaps(n_minutes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 3000.0
    log_index = 0
    for minute in range(n_minutes):
        # several swaps per minute, sharing a timestamp, to exercise tie ordering
        for _ in range(int(rng.integers(0, 4))):
            price *= 1 + rng.normal(0, 0.0006)
            log_index += 1
            rows.append(
                {
                    "block_number": 1_000 + minute // 12,
                    "log_index": log_index,
                    "timestamp": minute * 60,
                    "sqrt_price_x96": price_to_sqrt_price_x96(1.0 / price, decimals0=6, decimals1=18),
                    "liquidity": int(10**18 * rng.uniform(0.5, 2.0)),
                    "base_fee_wei": int(rng.integers(5, 40)) * 10**9,
                    "amount0": abs(rng.normal(5000, 1000)) * 10**6,
                }
            )
    # shuffled on purpose: the replay must not depend on input row order
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _assembled(seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    swaps = _swaps(4000, seed)
    bars = build_bars(swaps, interval_seconds=60)
    assembled = assemble_training_data(bars, swaps, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=0.005)
    return assembled, swaps


def test_replaying_a_bars_fills_at_the_reference_size_reproduces_its_label():
    assembled, swaps = _assembled()
    inputs = build_backtest_inputs(assembled, swaps, FEATURE_COLUMNS)

    assert len(inputs.entry_prices) == len(assembled) > 100
    for i in range(len(assembled)):
        pnl = simulate_fill(
            config.REFERENCE_NOTIONAL_USD, inputs.entry_prices[i], inputs.exit_prices[i], inputs.entry_swaps[i], inputs.exit_swaps[i]
        )
        assert abs(pnl / config.REFERENCE_NOTIONAL_USD - assembled["net_return"].iloc[i]) < 1e-9


def test_backtest_inputs_carry_raw_features_and_decision_timestamps():
    assembled, swaps = _assembled()
    inputs = build_backtest_inputs(assembled, swaps, FEATURE_COLUMNS)

    assert inputs.features.shape == (len(assembled), len(FEATURE_COLUMNS))
    np.testing.assert_allclose(inputs.features, assembled[FEATURE_COLUMNS].to_numpy(dtype="float64"))
    np.testing.assert_array_equal(inputs.timestamps, assembled["bar_start"].to_numpy())
    assert (np.diff(inputs.timestamps) >= 0).all()


def test_daily_sharpe_uses_end_of_day_equity_and_annualizes_over_365_days():
    from latentedge.backtest import daily_sharpe

    day = 86_400
    # end-of-day equity: 100 -> 110 -> 99 -> 108.9, so daily returns +10%, -10%, +10%
    equity = np.array([90.0, 100.0, 110.0, 99.0, 108.9])
    timestamps = np.array([0, day // 2, day, 2 * day, 3 * day])
    returns = np.array([0.10, -0.10, 0.10])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(365)
    assert abs(daily_sharpe(equity, timestamps) - expected) < 1e-9


def test_daily_sharpe_is_zero_when_there_is_nothing_to_measure():
    from latentedge.backtest import daily_sharpe

    day = 86_400
    assert daily_sharpe(np.array([100.0, 100.0]), np.array([0, day])) == 0.0  # zero variance
    assert daily_sharpe(np.array([100.0, 101.0]), np.array([0, 10])) == 0.0  # a single day
