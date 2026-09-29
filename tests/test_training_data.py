import numpy as np
import pandas as pd
import pytest

from latentedge.bars import build_bars
from latentedge.training_data import FEATURE_COLUMNS, assemble_training_data
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _random_walk_swaps(n_minutes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = 3000.0
    for minute in range(n_minutes):
        if rng.random() < 0.9:  # dense activity, most minutes have a swap
            price *= 1 + rng.normal(0, 0.0008)
            raw_price = 1.0 / price
            rows.append(
                {
                    "timestamp": minute * 60,
                    "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
                    "liquidity": 10**18,
                    "base_fee_wei": 15_000_000_000,
                    "amount0": abs(rng.normal(5000, 1000)) * 10**6,  # raw USDC units
                }
            )
    return pd.DataFrame(rows)


def test_assembled_training_data_has_no_lookahead_correlation_on_a_random_walk():
    # Regression test for a real leakage bug: a bar's own features
    # (return_1 in particular) were computed from data up to that bar's
    # close, but the label's entry fill executes at the bar's start —
    # earlier than the bar's own close. On a pure random walk, any real
    # predictive correlation is impossible by construction; a leak shows
    # up as return_1 correlating with net_return anyway.
    swaps = _random_walk_swaps(n_minutes=6000, seed=7)
    bars = build_bars(swaps, interval_seconds=60)

    assembled = assemble_training_data(
        bars, swaps, return_windows=[1, 5, 15, 30], volatility_window=15, tp_sl_fraction=0.01
    )

    assert len(assembled) > 100

    correlation = assembled["return_1"].corr(assembled["net_return"])
    assert abs(correlation) < 0.05, f"return_1 correlates with net_return at {correlation:.4f} — lookahead leak"


def test_assemble_training_data_returns_expected_columns():
    swaps = _random_walk_swaps(n_minutes=3000, seed=1)
    bars = build_bars(swaps, interval_seconds=60)
    assembled = assemble_training_data(bars, swaps, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=0.01)

    for column in FEATURE_COLUMNS + ["net_return", "gross_return"]:
        assert column in assembled.columns
    assert not assembled[FEATURE_COLUMNS].isna().any().any()
    assert not assembled["net_return"].isna().any()


def test_assemble_training_data_keeps_swap_indices_for_replay():
    swaps = _random_walk_swaps(n_minutes=3000, seed=1)
    bars = build_bars(swaps, interval_seconds=60)
    assembled = assemble_training_data(bars, swaps, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=0.01)

    assert (assembled["entry_swap_idx"] >= 0).all()
    assert (assembled["exit_swap_idx"] >= assembled["entry_swap_idx"]).all()
