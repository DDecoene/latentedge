from pathlib import Path

import mlx.core as mx
import mlx.utils as mlx_utils
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


def test_train_calls_on_epoch_with_progress_and_loss():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3)).astype(np.float32)
    y = rng.normal(size=(10,)).astype(np.float32)

    regressor = NetReturnRegressor(input_dim=3)

    calls: list[tuple[int, int, float]] = []

    def on_epoch(epoch: int, total_epochs: int, loss: float) -> None:
        calls.append((epoch, total_epochs, loss))

    train(regressor, x, y, epochs=3, learning_rate=0.001, on_epoch=on_epoch)

    assert [c[:2] for c in calls] == [(1, 3), (2, 3), (3, 3)]
    assert all(isinstance(c[2], float) for c in calls)


def test_a_deeper_network_saves_and_loads_with_its_own_shape(tmp_path):
    regressor = NetReturnRegressor(input_dim=3, hidden=(8, 4))
    path = tmp_path / "model.safetensors"
    save(regressor, path)

    loaded = load(path, input_dim=3)
    x = mx.array(np.random.default_rng(0).normal(size=(5, 3)).astype("float32"))

    assert np.allclose(np.array(loaded(x)), np.array(regressor(x)))


def test_a_single_layer_of_another_width_loads_and_the_old_default_is_unchanged(tmp_path):
    wide = NetReturnRegressor(input_dim=3, hidden=(32,))
    path = tmp_path / "wide.safetensors"
    save(wide, path)
    x = mx.array(np.ones((2, 3), dtype="float32"))

    assert np.allclose(np.array(load(path, input_dim=3)(x)), np.array(wide(x)))
    # models saved before the width was configurable used these names and 16 units
    assert {name for name, _ in mlx_utils.tree_flatten(NetReturnRegressor(input_dim=3).parameters())} == {
        "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias",
    }


def test_early_stopping_halts_when_validation_stops_improving_and_keeps_the_best_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 5)).astype("float32")
    y = rng.normal(size=60).astype("float32")  # pure noise: nothing that carries over
    val_x = rng.normal(size=(60, 5)).astype("float32")
    val_y = rng.normal(size=60).astype("float32")
    model = NetReturnRegressor(input_dim=5, hidden=(64, 64))

    losses = train(model, x, y, epochs=2000, learning_rate=0.01, validation=(val_x, val_y), patience=10)

    assert len(losses) < 2000
    best = float(np.mean((np.array(model(mx.array(val_x))) - val_y) ** 2))
    fresh = float(np.mean((np.array(NetReturnRegressor(input_dim=5, hidden=(64, 64))(mx.array(val_x))) - val_y) ** 2))
    assert best < fresh + 1.0  # restored to the best epoch, not left overfit at the end


def test_without_validation_training_runs_every_epoch():
    rng = np.random.default_rng(2)
    model = NetReturnRegressor(input_dim=2)
    losses = train(model, rng.normal(size=(20, 2)).astype("float32"), rng.normal(size=20).astype("float32"),
                   epochs=7, learning_rate=0.01)
    assert len(losses) == 7
