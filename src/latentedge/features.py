import json
from pathlib import Path

import numpy as np
import pandas as pd

# Order-flow features: who is pushing the pool, not just where the price is.
# Price bars keep only the last price and total volume; the signed dollar flow
# and the biggest single swap are what a price-only model never sees.
FLOW_WINDOWS = [5, 15, 30]
LARGE_SWAP_WINDOW = 15
ORDER_FLOW_COLUMNS = [f"flow_imbalance_{n}" for n in FLOW_WINDOWS] + [f"large_swap_share_{LARGE_SWAP_WINDOW}"]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, 0 where nothing traded (no flow, no imbalance)
    but NaN kept where the window is not yet full."""
    ratio = numerator / denominator.where(denominator > 0)
    return ratio.where(~(denominator == 0) | numerator.isna(), 0.0)


def compute_features(bars: pd.DataFrame, return_windows: list[int], volatility_window: int) -> pd.DataFrame:
    result = bars.copy()
    price = result["price_usdc_per_weth"]

    for n in return_windows:
        result[f"return_{n}"] = price.pct_change(periods=n)

    one_bar_return = price.pct_change(periods=1)
    result["volatility"] = one_bar_return.rolling(window=volatility_window, min_periods=volatility_window).std()

    if "net_flow_usdc" in result.columns:
        for n in FLOW_WINDOWS:
            flow = result["net_flow_usdc"].rolling(window=n, min_periods=n).sum()
            volume = result["volume_usdc"].rolling(window=n, min_periods=n).sum()
            # In [-1, 1]: +1 is all buying over the window, -1 all selling.
            result[f"flow_imbalance_{n}"] = _safe_ratio(flow, volume)
        biggest = result["max_swap_usdc"].rolling(window=LARGE_SWAP_WINDOW, min_periods=LARGE_SWAP_WINDOW).max()
        volume = result["volume_usdc"].rolling(window=LARGE_SWAP_WINDOW, min_periods=LARGE_SWAP_WINDOW).sum()
        # How much of the window's volume one swap accounts for: high when a
        # single large trader, not many small ones, is moving the pool.
        result[f"large_swap_share_{LARGE_SWAP_WINDOW}"] = _safe_ratio(biggest, volume)

    had_swap = result["swap_count"] > 0
    groups = had_swap.cumsum()
    result["bars_since_swap"] = result.groupby(groups).cumcount()
    result.loc[had_swap, "bars_since_swap"] = 0

    return result


def shift_features_for_labeling(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Shift feature columns back by one bar so a label at row t pairs
    only with data known strictly before that bar began.

    A bar's own features (return_n, volatility, ...) reflect swap data up
    to and including that bar's close, but the triple-barrier label at
    the same bar enters at the bar's *start* — the first swap at or after
    t, which can be earlier than the bar's own close. Pairing a label
    with its own bar's features leaks up to one bar's worth of
    post-entry price action into the inputs. Shifting closes that gap.
    """
    result = df.copy()
    result[feature_columns] = result[feature_columns].shift(1)
    return result


def compute_feature_stats(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, tuple[float, float]]:
    """Per-column (mean, std) computed from a training split, for later
    standardization at both train and inference time. Must be computed
    from the train split only — computing from validate/test data would
    leak information about those splits into training."""
    stats: dict[str, tuple[float, float]] = {}
    for column in feature_columns:
        mean = float(df[column].mean())
        std = float(df[column].std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[column] = (mean, std)
    return stats


def standardize_features(df: pd.DataFrame, feature_columns: list[str], stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Z-score each feature column using precomputed (mean, std) stats.

    Without this, features on wildly different scales (e.g. a ~1e10
    volume figure alongside a ~1e-2 return figure) make gradient descent
    effectively ignore the small-scale features — verified empirically to
    produce a model that predicts a constant, absurdly large value
    regardless of input.
    """
    result = df.copy()
    for column in feature_columns:
        mean, std = stats[column]
        result[column] = (result[column] - mean) / std
    return result


def standardize_value(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    """Z-score a single array (e.g. a regression target) using a
    precomputed (mean, std) pair — the array counterpart to
    standardize_features for a column that isn't part of a DataFrame."""
    mean, std = stats
    return (values - mean) / std


def unstandardize_value(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    """Inverse of standardize_value — maps a model's standardized-scale
    output back to the original units for reporting/consumption."""
    mean, std = stats
    return values * std + mean


def target_stats(stats: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """(mean, std) of whatever the model was trained to predict: the "target"
    entry, or net_return for stats saved before the target was selectable."""
    return stats["target"] if "target" in stats else stats["net_return"]


def save_feature_stats(stats: dict[str, tuple[float, float]], path: Path) -> None:
    path.write_text(json.dumps(stats))


def load_feature_stats(path: Path) -> dict[str, tuple[float, float]]:
    raw: dict[str, list[float]] = json.loads(path.read_text())
    return {column: (values[0], values[1]) for column, values in raw.items()}


def stats_to_arrays(stats: dict[str, tuple[float, float]], feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Stats as parallel (means, stds) arrays in feature_columns order —
    for standardizing a raw numpy feature matrix (SignalClient's input
    shape) rather than a named DataFrame."""
    means = np.array([stats[column][0] for column in feature_columns])
    stds = np.array([stats[column][1] for column in feature_columns])
    return means, stds
