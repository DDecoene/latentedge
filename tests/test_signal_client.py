from pathlib import Path

import numpy as np
import pytest

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
