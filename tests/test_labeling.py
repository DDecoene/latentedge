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


def test_label_bars_reports_progress_and_can_be_cancelled():
    swaps = pd.DataFrame([_swap(t, 3000.0) for t in range(0, 4000, 10)])
    bars = pd.DataFrame(
        [{"bar_start": t, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1.0, "has_gap": False}
         for t in range(0, 2000, 60)]
    )
    seen: list[tuple[int, int]] = []
    label_bars(bars, swaps, tp_sl_fraction=0.01, on_progress=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1] == (len(bars), len(bars))

    class Stop(Exception):
        pass

    def cancel(done: int, total: int) -> None:
        raise Stop

    with pytest.raises(Stop):
        label_bars(bars, swaps, tp_sl_fraction=0.01, on_progress=cancel)


def test_labeled_bar_reports_the_swap_rows_it_entered_and_exited_on():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(30, 3001.0), _swap(60, 3100.0)])
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 3, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    # entry is the first swap at/after the bar; the 1% barrier is first crossed by the third swap
    assert result.iloc[0]["entry_swap_idx"] == 0
    assert result.iloc[0]["exit_swap_idx"] == 2


def test_excluded_bar_has_no_swap_indices():
    swaps = pd.DataFrame([_swap(2000, 3000.0)])
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 0, "volume_usdc": 0.0, "has_gap": True}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["entry_swap_idx"] == -1
    assert result.iloc[0]["exit_swap_idx"] == -1


def test_sort_swaps_is_stable_for_swaps_sharing_a_timestamp():
    from latentedge.labeling import sort_swaps

    swaps = pd.DataFrame(
        {"timestamp": [5, 1, 5, 1, 5], "block_number": [2, 1, 2, 1, 2], "log_index": [9, 4, 3, 2, 1], "marker": list("abcde")}
    )
    ordered = sort_swaps(swaps)
    # ties broken by (block_number, log_index) when present — never arbitrary
    assert ordered["marker"].tolist() == ["d", "b", "e", "c", "a"]


def test_gross_return_is_the_price_move_before_any_cost():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(60, 3100.0)])
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)

    assert result.iloc[0]["gross_return"] == pytest.approx(100.0 / 3000.0)
    assert result.iloc[0]["net_return"] < result.iloc[0]["gross_return"]


def test_excluded_bar_has_no_gross_return():
    swaps = pd.DataFrame([_swap(2000, 3000.0)])
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 0, "volume_usdc": 0.0, "has_gap": True}])
    assert pd.isna(label_bars(bars, swaps, tp_sl_fraction=0.01).iloc[0]["gross_return"])


def test_a_longer_horizon_keeps_a_bar_open_until_a_later_barrier_is_hit():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(600, 3001.0), _swap(3000, 3100.0), _swap(20000, 3100.0)])
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])

    short = label_bars(bars, swaps, tp_sl_fraction=0.01)
    long = label_bars(bars, swaps, tp_sl_fraction=0.01, horizon_seconds=4 * 3600)

    assert short.iloc[0]["exit_swap_idx"] == 1  # 30 minutes: no barrier, exits at the last swap in the window
    assert long.iloc[0]["exit_swap_idx"] == 2  # 4 hours: the 3.3% move at t=3000 is inside the horizon
