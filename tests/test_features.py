import pandas as pd
import pytest

from latentedge.features import compute_features


def test_features_use_only_past_data():
    bars = pd.DataFrame(
        {
            "bar_start": [0, 60, 120, 180, 240],
            "price_usdc_per_weth": [3000.0, 3010.0, 3005.0, 3020.0, 3015.0],
            "volume_usdc": [100.0, 200.0, 150.0, 300.0, 250.0],
            "swap_count": [1, 1, 0, 1, 1],
        }
    )
    result = compute_features(bars, return_windows=[2], volatility_window=3)

    # return_2 at index 2 uses prices at index 0 and 2 only, not index 3/4.
    expected_return_2_at_idx2 = (3005.0 - 3000.0) / 3000.0
    assert result.loc[2, "return_2"] == pytest.approx(expected_return_2_at_idx2)

    # Changing a *future* price must not change a past feature value.
    bars_altered = bars.copy()
    bars_altered.loc[4, "price_usdc_per_weth"] = 99999.0
    result_altered = compute_features(bars_altered, return_windows=[2], volatility_window=3)
    assert result_altered.loc[2, "return_2"] == result.loc[2, "return_2"]


def test_bars_since_swap_resets_on_activity():
    bars = pd.DataFrame(
        {
            "bar_start": [0, 60, 120, 180],
            "price_usdc_per_weth": [3000.0, 3000.0, 3000.0, 3000.0],
            "volume_usdc": [100.0, 0.0, 0.0, 50.0],
            "swap_count": [1, 0, 0, 1],
        }
    )
    result = compute_features(bars, return_windows=[1], volatility_window=2)
    assert list(result["bars_since_swap"]) == [0, 1, 2, 0]
