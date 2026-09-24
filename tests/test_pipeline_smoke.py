from pathlib import Path

import numpy as np
import pandas as pd

from latentedge.bars import build_bars
from latentedge.features import compute_features
from latentedge.labeling import label_bars
from latentedge.model import NetReturnRegressor, save, train
from latentedge.signal_client import SignalClient
from latentedge.split import chronological_split
from latentedge.uniswap_math import price_to_sqrt_price_x96

FEATURE_COLUMNS = ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap"]


def _synthetic_swaps(n_minutes: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    price = 3000.0
    for minute in range(n_minutes):
        if rng.random() < 0.7:  # most minutes have at least one swap
            price *= 1 + rng.normal(0, 0.001)
            # See Task 2's ruling: price_to_sqrt_price_x96 already
            # decimal-adjusts internally, so pass 1.0 / price directly.
            raw_price = 1.0 / price
            rows.append(
                {
                    "timestamp": minute * 60,
                    "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
                    "liquidity": 10**18,
                    "base_fee_wei": 20_000_000_000,
                    "amount0": abs(rng.normal(1000, 200)),
                }
            )
    return pd.DataFrame(rows)


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path: Path):
    swaps = _synthetic_swaps(n_minutes=5000)

    bars = build_bars(swaps, interval_seconds=60)
    labeled = label_bars(bars, swaps, tp_sl_fraction=0.01)
    labeled = labeled[~labeled["excluded"]].reset_index(drop=True)
    assert len(labeled) > 0

    featured = compute_features(labeled, return_windows=[5, 15, 30], volatility_window=15)
    # A bare dropna() drops on the "reason" column too, which is legitimately
    # None for every non-excluded row (only excluded rows get a reason
    # string) — that wipes the entire frame. Target the columns that
    # actually matter for training instead.
    featured = featured.dropna(subset=FEATURE_COLUMNS + ["net_return"]).reset_index(drop=True)
    assert len(featured) > 100

    train_split, validate_split, test_split = chronological_split(featured, train_fraction=0.7, validate_fraction=0.15)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    regressor = NetReturnRegressor(input_dim=len(FEATURE_COLUMNS))
    train(regressor, x, y, epochs=10, learning_rate=0.001)

    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=len(FEATURE_COLUMNS))
    test_features = test_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    predictions = client.predict_batch(test_features)

    assert predictions.shape == (len(test_split),)
    assert np.isfinite(predictions).all()
