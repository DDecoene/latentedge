from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latentedge.features import (
    compute_feature_stats,
    compute_features,
    load_feature_stats,
    save_feature_stats,
    shift_features_for_labeling,
    standardize_features,
    standardize_value,
    stats_to_arrays,
    unstandardize_value,
)


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


def test_shift_features_for_labeling_pairs_row_t_with_features_from_t_minus_1():
    # Regression test: a bar's own features are computed from data up to
    # and including that bar's close, but the triple-barrier label at the
    # same bar enters at the bar's *start* (the first swap at/after t) —
    # which can be earlier than the bar's own close. Pairing a label with
    # its own bar's features leaks up to one bar's worth of post-entry
    # data. Shifting closes that gap: a label at row t must pair only
    # with features known strictly before that bar began, i.e. row t-1's.
    df = pd.DataFrame({"feature_a": [10.0, 20.0, 30.0, 40.0], "net_return": [0.01, 0.02, 0.03, 0.04]})
    result = shift_features_for_labeling(df, feature_columns=["feature_a"])

    assert pd.isna(result.loc[0, "feature_a"])  # no prior bar to shift in from
    assert result.loc[1, "feature_a"] == 10.0  # row 1's feature is row 0's original value
    assert result.loc[2, "feature_a"] == 20.0
    assert result.loc[3, "feature_a"] == 30.0
    # net_return is untouched — only feature columns shift.
    assert list(result["net_return"]) == [0.01, 0.02, 0.03, 0.04]


def test_standardize_features_produces_zero_mean_unit_std_on_its_own_stats():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [100.0, 200.0, 300.0, 400.0, 500.0]})
    stats = compute_feature_stats(df, feature_columns=["a", "b"])
    standardized = standardize_features(df, feature_columns=["a", "b"], stats=stats)

    assert standardized["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert standardized["a"].std() == pytest.approx(1.0, rel=1e-6)
    assert standardized["b"].mean() == pytest.approx(0.0, abs=1e-9)
    assert standardized["b"].std() == pytest.approx(1.0, rel=1e-6)


def test_standardize_features_closes_a_1000x_scale_gap():
    # Regression test for a real bug: an unnormalized volume feature on
    # the order of 1e10 alongside return features on the order of 1e-2
    # made training produce garbage (verified empirically: loss stuck at
    # 1.3e13, predictions up to $7.9M). After standardization, both
    # features must be on comparable scales.
    df = pd.DataFrame({"tiny_return": [0.01, -0.02, 0.015, -0.005, 0.03], "huge_volume": [1e10, 2e10, 1.5e10, 1.8e10, 1.2e10]})
    stats = compute_feature_stats(df, feature_columns=["tiny_return", "huge_volume"])
    standardized = standardize_features(df, feature_columns=["tiny_return", "huge_volume"], stats=stats)

    assert standardized["tiny_return"].abs().max() < 5.0
    assert standardized["huge_volume"].abs().max() < 5.0


def test_standardize_features_handles_zero_variance_column_without_dividing_by_zero():
    df = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "varying": [1.0, 2.0, 3.0]})
    stats = compute_feature_stats(df, feature_columns=["constant", "varying"])
    standardized = standardize_features(df, feature_columns=["constant", "varying"], stats=stats)

    assert np.isfinite(standardized["constant"]).all()
    assert (standardized["constant"] == 0.0).all()  # (5 - 5) / 1.0 fallback std


def test_feature_stats_round_trip_through_disk(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    stats = compute_feature_stats(df, feature_columns=["a", "b"])

    path = tmp_path / "stats.json"
    save_feature_stats(stats, path)
    loaded = load_feature_stats(path)

    assert loaded == stats


def test_stats_to_arrays_preserves_feature_column_order():
    stats = {"a": (1.0, 2.0), "b": (3.0, 4.0), "c": (5.0, 6.0)}
    means, stds = stats_to_arrays(stats, feature_columns=["c", "a", "b"])
    assert list(means) == [5.0, 1.0, 3.0]
    assert list(stds) == [6.0, 2.0, 4.0]


def test_standardize_value_produces_zero_mean_unit_std():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = (float(values.mean()), float(values.std()))
    standardized = standardize_value(values, stats)

    assert standardized.mean() == pytest.approx(0.0, abs=1e-9)
    assert standardized.std() == pytest.approx(1.0, rel=1e-6)


def test_unstandardize_value_inverts_standardize_value():
    values = np.array([0.001, -0.002, 0.0015, -0.0005, 0.003])
    stats = (float(values.mean()), float(values.std()))

    round_tripped = unstandardize_value(standardize_value(values, stats), stats)

    assert np.allclose(round_tripped, values)


def test_standardize_array_matches_standardize_features():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    stats = compute_feature_stats(df, feature_columns=["a", "b"])
    expected = standardize_features(df, feature_columns=["a", "b"], stats=stats)[["a", "b"]].to_numpy()

    means, stds = stats_to_arrays(stats, feature_columns=["a", "b"])
    raw = df[["a", "b"]].to_numpy()
    actual = (raw - means) / stds

    assert np.allclose(actual, expected)


def _flow_bars(net_flow, volume, biggest):
    n = len(net_flow)
    return pd.DataFrame(
        {
            "bar_start": [60 * i for i in range(n)],
            "price_usdc_per_weth": [3000.0] * n,
            "volume_usdc": volume, "net_flow_usdc": net_flow, "max_swap_usdc": biggest, "swap_count": [1] * n,
        }
    )


def test_flow_imbalance_is_net_buying_share_of_volume_over_the_window():
    bars = _flow_bars([100.0] * 5 + [-100.0] * 5, [100.0] * 10, [100.0] * 10)
    result = compute_features(bars, return_windows=[2], volatility_window=3)

    assert result.loc[4, "flow_imbalance_5"] == pytest.approx(1.0)   # five buy bars
    assert result.loc[9, "flow_imbalance_5"] == pytest.approx(-1.0)  # five sell bars
    assert pd.isna(result.loc[3, "flow_imbalance_5"])                # window not yet full


def test_flow_features_use_only_past_data():
    net_flow, volume, biggest = [50.0] * 40, [100.0] * 40, [60.0] * 40
    bars = _flow_bars(net_flow, volume, biggest)
    altered = _flow_bars(net_flow[:35] + [-9999.0] * 5, volume[:35] + [9999.0] * 5, biggest[:35] + [9999.0] * 5)

    base = compute_features(bars, return_windows=[2], volatility_window=3)
    changed = compute_features(altered, return_windows=[2], volatility_window=3)

    for column in ("flow_imbalance_5", "flow_imbalance_30", "large_swap_share_15"):
        assert changed.loc[34, column] == base.loc[34, column]


def test_large_swap_share_and_a_window_with_no_trades():
    bars = _flow_bars([0.0] * 20, [10.0] * 10 + [0.0] * 10, [4.0] * 10 + [0.0] * 10)
    result = compute_features(bars, return_windows=[2], volatility_window=3)

    assert result.loc[14, "large_swap_share_15"] == pytest.approx(4.0 / 100.0)
    assert result.loc[19, "flow_imbalance_5"] == 0.0  # nothing traded: no imbalance, not NaN
