import pandas as pd
import pytest

from latentedge.bars import build_bars
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap(timestamp: int, human_price: float, amount0: float) -> dict:
    # price_to_sqrt_price_x96 already decimal-adjusts internally, so the
    # human-terms WETH-per-USDC price (inverse of USDC-per-WETH) is
    # passed directly, with no additional scaling (see Task 2's ruling on
    # this exact double-scaling mistake).
    raw_price = 1.0 / human_price
    return {
        "timestamp": timestamp,
        "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
        "amount0": amount0,
        "amount1": -amount0 / human_price,
    }


def test_bar_with_no_swaps_carries_price_forward_and_flags_gap():
    swaps = pd.DataFrame([_swap(0, 3000.0, 1000.0), _swap(150, 3010.0, 500.0)])
    bars = build_bars(swaps, interval_seconds=60)

    # bar[0] = [0,60): one swap at t=0
    # bar[1] = [60,120): no swaps -> carried forward from bar[0]
    # bar[2] = [120,180): one swap at t=150
    assert len(bars) == 3
    assert bars.iloc[1]["swap_count"] == 0
    assert bars.iloc[1]["has_gap"]
    assert bars.iloc[1]["price_usdc_per_weth"] == pytest.approx(bars.iloc[0]["price_usdc_per_weth"], rel=1e-6)
    assert not bars.iloc[2]["has_gap"]


def test_bar_with_many_swaps_aggregates_volume():
    swaps = pd.DataFrame([_swap(0, 3000.0, 100.0), _swap(10, 3001.0, 200.0), _swap(20, 3002.0, 50.0)])
    bars = build_bars(swaps, interval_seconds=60)
    assert len(bars) == 1
    assert bars.iloc[0]["swap_count"] == 3
    assert bars.iloc[0]["volume_usdc"] == pytest.approx(350.0)
