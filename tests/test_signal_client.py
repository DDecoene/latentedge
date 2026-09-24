from pathlib import Path

import numpy as np
import pytest

from latentedge.features import save_feature_stats
from latentedge.model import NetReturnRegressor, save, train
from latentedge.signal_client import PredictionError, SignalClient


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.normal(size=(100,)).astype(np.float32)
    regressor = NetReturnRegressor(input_dim=3)
    train(regressor, x, y, epochs=5, learning_rate=0.01)
    path = tmp_path / "model.safetensors"
    save(regressor, path)
    return path


def test_predict_batch_runs_real_forward_pass(trained_model_path: Path):
    client = SignalClient(trained_model_path, input_dim=3)
    features = np.random.default_rng(3).normal(size=(10, 3)).astype(np.float32)
    predictions = client.predict_batch(features)
    assert predictions.shape == (10,)
    assert not np.isnan(predictions).any()


def test_predict_batch_rejects_nan_input(trained_model_path: Path):
    client = SignalClient(trained_model_path, input_dim=3)
    features = np.array([[1.0, float("nan"), 3.0]], dtype=np.float32)
    with pytest.raises(PredictionError):
        client.predict_batch(features)


def test_predict_batch_standardizes_using_saved_stats_when_present(trained_model_path: Path):
    # Regression test: without standardization, a wildly-scaled raw
    # feature (e.g. real raw-unit volume, ~1e10) dominates a well-scaled
    # one and the model effectively ignores the small feature. If a
    # stats sidecar exists next to the model, predict_batch must apply
    # it before calling the model — verified here by confirming the
    # client actually reads and uses the stats rather than ignoring them.
    feature_columns = ["a", "b", "c"]
    stats = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (1e10, 1e9)}
    stats_path = Path(str(trained_model_path) + ".stats.json")
    save_feature_stats(stats, stats_path)

    client = SignalClient(trained_model_path, input_dim=3, feature_columns=feature_columns)

    # A raw feature row whose 3rd column is on the ~1e10 scale the stats
    # describe. If standardization is applied, this becomes a small,
    # finite value fed to the model; if not, the model sees ~1e10
    # directly. Either way predict_batch must not raise, and the
    # standardized path must produce a different result than an
    # unstandardized client would for the same raw input.
    raw_features = np.array([[0.5, -0.3, 1.05e10]], dtype=np.float32)
    standardized_prediction = client.predict_batch(raw_features)

    unstandardized_client = SignalClient(trained_model_path, input_dim=3)
    unstandardized_prediction = unstandardized_client.predict_batch(raw_features)

    assert not np.isnan(standardized_prediction).any()
    assert not np.allclose(standardized_prediction, unstandardized_prediction)
