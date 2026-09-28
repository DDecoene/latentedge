import pandas as pd
import pytest

from latentedge.bars import build_bars
from latentedge.uniswap_math import price_to_sqrt_price_x96

USDC_DECIMALS = 6
USDC_SCALE = 10**USDC_DECIMALS


def _swap(timestamp: int, human_price: float, human_amount0_usdc: float, base_fee_gwei: float = 20.0) -> dict:
    # price_to_sqrt_price_x96 already decimal-adjusts internally, so the
    # human-terms WETH-per-USDC price (inverse of USDC-per-WETH) is
    # passed directly, with no additional scaling (see Task 2's ruling on
    # this exact double-scaling mistake).
    #
    # amount0 as decoded from a real Swap event is in RAW token0 units
    # (USDC has 6 decimals) — e.g. a real ~$11,923 swap decodes to
    # amount0=11_923_388_581, not amount0=11923.0 (confirmed against the
    # live RPC data in Task 6). Fixtures here build from a human dollar
    # figure and scale up, so tests exercise the same raw-unit shape real
    # ingestion produces.
    raw_price = 1.0 / human_price
    raw_amount0 = human_amount0_usdc * USDC_SCALE
    return {
        "timestamp": timestamp,
        "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
        "amount0": raw_amount0,
        "amount1": -raw_amount0 / human_price,
        "base_fee_wei": int(base_fee_gwei * 1_000_000_000),
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


def test_base_fee_gwei_carries_forward_through_gap_bars():
    swaps = pd.DataFrame([_swap(0, 3000.0, 1000.0, base_fee_gwei=15.0), _swap(150, 3010.0, 500.0, base_fee_gwei=25.0)])
    bars = build_bars(swaps, interval_seconds=60)

    assert bars.iloc[0]["base_fee_gwei"] == pytest.approx(15.0)
    assert bars.iloc[1]["base_fee_gwei"] == pytest.approx(15.0)  # gap bar carries the last known fee forward
    assert bars.iloc[2]["base_fee_gwei"] == pytest.approx(25.0)


def test_volume_usdc_is_decimal_adjusted_from_raw_amount0():
    # Regression test for a real bug: amount0 is decoded from the raw
    # Swap event in USDC's 6-decimal raw units, not human dollars. A bare
    # `abs(amount0)` treated as volume_usdc would be 1,000,000x too large.
    swaps = pd.DataFrame([_swap(0, 3000.0, 11_923.388581)])
    bars = build_bars(swaps, interval_seconds=60)
    assert bars.iloc[0]["volume_usdc"] == pytest.approx(11_923.388581, rel=1e-6)
