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

from latentedge.features import ORDER_FLOW_COLUMNS, compute_features, shift_features_for_labeling
from latentedge.labeling import LabelSettings, label_bars

PRICE_FEATURE_COLUMNS = [
    "return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap", "base_fee_gwei",
]
FEATURE_COLUMNS = [*PRICE_FEATURE_COLUMNS, *ORDER_FLOW_COLUMNS]

# Names an exclusion setting may use for a set of features at once.
FEATURE_GROUPS = {
    "order_flow": ORDER_FLOW_COLUMNS,
    "cost": ["base_fee_gwei", "volatility"],
    "returns": ["return_5", "return_15", "return_30"],
}


class SplitArrays(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    # Raw (unstandardized) pre-cost return of each row's round trip, for
    # checking whether predictions track direction or only trading cost.
    gross: np.ndarray | None = None


class AssembledTrainingData(NamedTuple):
    train: SplitArrays
    validate: SplitArrays
    test: SplitArrays
    input_dim: int
    stats: dict[str, tuple[float, float]]
    # bar_start of the first test-split bar: where the untouched test
    # window begins. Recorded with the model so a later backtest replays
    # only bars the model never trained or validated on.
    test_start: int | None = None
    # The columns x holds, in order, and how the labels were built — saved
    # with the model so a later backtest reproduces both.
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    label_settings: LabelSettings = LabelSettings()


def assemble_training_data(
    bars: pd.DataFrame,
    swaps: pd.DataFrame,
    return_windows: list[int],
    volatility_window: int,
    tp_sl_fraction: float,
    on_label_progress: Callable[[int, int], None] | None = None,
    horizon_seconds: int = LabelSettings().horizon_seconds,
) -> pd.DataFrame:
    feature_columns = [f"return_{n}" for n in return_windows] + [
        "volatility", "volume_usdc", "bars_since_swap", "base_fee_gwei", *ORDER_FLOW_COLUMNS,
    ]

    # Compute features and labels on the same full, contiguous bar
    # series (not a post-exclusion-filtered one) so rolling windows never
    # silently span a gap where excluded rows were removed.
    featured = compute_features(bars, return_windows=return_windows, volatility_window=volatility_window)
    labeled = label_bars(
        bars, swaps, tp_sl_fraction=tp_sl_fraction, on_progress=on_label_progress, horizon_seconds=horizon_seconds
    )

    combined = featured.copy()
    combined["net_return"] = labeled["net_return"]
    combined["gross_return"] = labeled["gross_return"]
    combined["excluded"] = labeled["excluded"]
    combined["reason"] = labeled["reason"]
    combined["entry_swap_idx"] = labeled["entry_swap_idx"]
    combined["exit_swap_idx"] = labeled["exit_swap_idx"]

    # Pair each label with the *previous* bar's features — see
    # shift_features_for_labeling's docstring for why bar t's own
    # features leak post-entry information into bar t's label.
    combined = shift_features_for_labeling(combined, feature_columns)

    combined = combined[~combined["excluded"]]
    combined = combined.dropna(subset=feature_columns + ["net_return"]).reset_index(drop=True)
    return combined
