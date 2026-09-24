from pathlib import Path

import numpy as np

from latentedge.model import NetReturnRegressor, load, save, train


def test_training_reduces_loss_on_learnable_synthetic_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    true_weights = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
    y = (x @ true_weights).astype(np.float32)

    regressor = NetReturnRegressor(input_dim=4)
    losses = train(regressor, x, y, epochs=50, learning_rate=0.05)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1  # should fit this simple linear signal closely


def test_save_and_load_round_trip_predicts_identically(tmp_path: Path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)).astype(np.float32)
    y = rng.normal(size=(50,)).astype(np.float32)

    regressor = NetReturnRegressor(input_dim=3)
    train(regressor, x, y, epochs=5, learning_rate=0.01)

    path = tmp_path / "model.safetensors"
    save(regressor, path)
    loaded = load(path, input_dim=3)

    import mlx.core as mx

    original_pred = regressor(mx.array(x))
    loaded_pred = loaded(mx.array(x))
    assert np.allclose(np.array(original_pred), np.array(loaded_pred), atol=1e-6)
