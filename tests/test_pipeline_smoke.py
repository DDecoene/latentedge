from pathlib import Path

import numpy as np
import pandas as pd

from latentedge.bars import build_bars
from latentedge.features import compute_feature_stats, save_feature_stats, standardize_features
from latentedge.model import NetReturnRegressor, save, train
from latentedge.signal_client import SignalClient
from latentedge.split import chronological_split
from latentedge.training_data import FEATURE_COLUMNS, assemble_training_data
from latentedge.uniswap_math import price_to_sqrt_price_x96

USDC_SCALE = 10**6  # amount0 is raw USDC (6-decimal) units, not human dollars


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
                    "amount0": abs(rng.normal(1000, 200)) * USDC_SCALE,
                }
            )
    return pd.DataFrame(rows)


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path: Path):
    # Mirrors the CLI's train command exactly — this test exists to catch
    # integration seams the per-module tests can't see, so it must run
    # the same assembly path (point-in-time feature/label pairing,
    # standardization) the real pipeline uses, not a simplified version
    # of it.
    swaps = _synthetic_swaps(n_minutes=5000)
    bars = build_bars(swaps, interval_seconds=60)

    assembled = assemble_training_data(bars, swaps, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=0.01)
    assert len(assembled) > 100

    train_split, validate_split, test_split = chronological_split(assembled, train_fraction=0.7, validate_fraction=0.15)

    stats = compute_feature_stats(train_split, FEATURE_COLUMNS)
    train_split = standardize_features(train_split, FEATURE_COLUMNS, stats)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    regressor = NetReturnRegressor(input_dim=len(FEATURE_COLUMNS))
    train(regressor, x, y, epochs=10, learning_rate=0.001)

    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)
    save_feature_stats(stats, Path(str(model_path) + ".stats.json"))

    client = SignalClient(model_path, input_dim=len(FEATURE_COLUMNS), feature_columns=FEATURE_COLUMNS)
    # Feed raw (unstandardized) features — SignalClient applies the saved
    # stats itself, the same way real inference would.
    raw_test_features = test_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    predictions = client.predict_batch(raw_test_features)

    assert predictions.shape == (len(test_split),)
    assert np.isfinite(predictions).all()
