import pandas as pd
import pytest

from latentedge.labeling import label_bars
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap(timestamp: int, human_price: float, liquidity: int = 10**18, base_fee_wei: int = 20_000_000_000) -> dict:
    # See Task 2's ruling: price_to_sqrt_price_x96 already decimal-adjusts
    # internally, so pass 1.0 / human_price directly, no extra scaling.
    raw_price = 1.0 / human_price
    return {
        "timestamp": timestamp,
        "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
        "liquidity": liquidity,
        "base_fee_wei": base_fee_wei,
    }


def test_bar_with_no_swap_in_window_is_excluded_for_no_entry_fill():
    # bar at t=0 has no swap at/after t=0 before the horizon ends at t=1800
    swaps = pd.DataFrame([_swap(2000, 3000.0)])  # only swap is after the horizon
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 0, "volume_usdc": 0.0, "has_gap": True}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["excluded"]
    assert result.iloc[0]["reason"] == "no_entry_fill"
    assert pd.isna(result.iloc[0]["net_return"])


def test_take_profit_hit_produces_positive_net_return():
    swaps = pd.DataFrame(
        [
            _swap(0, 3000.0),  # entry fill
            _swap(60, 3100.0),  # +3.3%, above a 1% take-profit band -> exit here
        ]
    )
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert not result.iloc[0]["excluded"]
    assert result.iloc[0]["net_return"] > 0
    # net return must be less than the raw 3.3% move once fee/slippage/gas are netted out
    assert result.iloc[0]["net_return"] < 0.033


def test_stop_loss_hit_produces_negative_net_return():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(60, 2900.0)])  # -3.3% move
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["net_return"] < 0


def test_bar_near_end_of_history_with_incomplete_horizon_is_excluded():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(60, 3001.0)])  # history ends at t=60, horizon needs t=1800
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["excluded"]
    assert result.iloc[0]["reason"] == "incomplete_horizon"
