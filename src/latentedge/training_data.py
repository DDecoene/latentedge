"""Assembles point-in-time-correct training data from bars and swaps.

This is the one place feature computation, labeling, and the
lookahead-leakage fix (shifting features back one bar) come together —
callers (the CLI, the smoke test) should use assemble_training_data
rather than composing features.compute_features and labeling.label_bars
by hand, since doing that separately and filtering before computing
features is exactly the bug this module exists to prevent.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from latentedge.features import compute_features, shift_features_for_labeling
from latentedge.labeling import label_bars

FEATURE_COLUMNS = ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap", "base_fee_gwei"]


class SplitArrays(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class AssembledTrainingData(NamedTuple):
    train: SplitArrays
    validate: SplitArrays
    test: SplitArrays
    input_dim: int
    stats: dict[str, tuple[float, float]]


def assemble_training_data(
    bars: pd.DataFrame,
    swaps: pd.DataFrame,
    return_windows: list[int],
    volatility_window: int,
    tp_sl_fraction: float,
    on_label_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    feature_columns = [f"return_{n}" for n in return_windows] + [
        "volatility", "volume_usdc", "bars_since_swap", "base_fee_gwei",
    ]

    # Compute features and labels on the same full, contiguous bar
    # series (not a post-exclusion-filtered one) so rolling windows never
    # silently span a gap where excluded rows were removed.
    featured = compute_features(bars, return_windows=return_windows, volatility_window=volatility_window)
    labeled = label_bars(bars, swaps, tp_sl_fraction=tp_sl_fraction, on_progress=on_label_progress)

    combined = featured.copy()
    combined["net_return"] = labeled["net_return"]
    combined["excluded"] = labeled["excluded"]
    combined["reason"] = labeled["reason"]

    # Pair each label with the *previous* bar's features — see
    # shift_features_for_labeling's docstring for why bar t's own
    # features leak post-entry information into bar t's label.
    combined = shift_features_for_labeling(combined, feature_columns)

    combined = combined[~combined["excluded"]]
    combined = combined.dropna(subset=feature_columns + ["net_return"]).reset_index(drop=True)
    return combined
