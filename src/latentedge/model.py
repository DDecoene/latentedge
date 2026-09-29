from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np


DEFAULT_HIDDEN = (16,)
EARLY_STOPPING_PATIENCE = 50


class NetReturnRegressor(nn.Module):  # type: ignore[misc]  # mlx ships no type stubs
    """A small fully connected network: one ReLU layer per entry of `hidden`,
    then a single output. One hidden layer keeps the original weight names
    (layer1, layer2) so models saved before the size was configurable still
    load; more layers are stored as a list."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = DEFAULT_HIDDEN):
        super().__init__()
        if len(hidden) == 1:
            self.layer1 = nn.Linear(input_dim, hidden[0])
            self.layer2 = nn.Linear(hidden[0], 1)
        else:
            sizes = [input_dim, *hidden, 1]
            self.layers = [nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x: mx.array) -> mx.array:
        if hasattr(self, "layer1"):
            h = nn.relu(self.layer1(x))
            result: mx.array = self.layer2(h).squeeze(-1)
            return result
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        result = self.layers[-1](x).squeeze(-1)
        return result


def _loss_fn(model: NetReturnRegressor, x: mx.array, y: mx.array) -> mx.array:
    predictions = model(x)
    return mx.mean((predictions - y) ** 2)


def train(
    model: NetReturnRegressor,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float,
    on_epoch: Callable[[int, int, float], None] | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> list[float]:
    """Full-batch Adam. With validation data, training stops once the loss on
    it has not improved for `patience` epochs and the weights from the best
    epoch are restored: a bigger network keeps fitting its training data long
    after it has stopped learning anything that carries over, and would be
    judged on the memorized part. (This uses the validation window to choose
    the stopping point; the test window stays untouched.) Returns the
    training loss of each epoch run."""
    x = mx.array(features)
    y = mx.array(labels)
    val_x, val_y = (mx.array(validation[0]), mx.array(validation[1])) if validation is not None else (None, None)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    losses: list[float] = []
    best_val = float("inf")
    best_params = None
    best_epoch = 0
    for epoch in range(epochs):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_value = float(loss)
        losses.append(loss_value)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs, loss_value)
        if val_x is not None:
            val_loss = float(_loss_fn(model, val_x, val_y))
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                best_params = tree_map(lambda a: mx.array(a), model.parameters())
            elif epoch - best_epoch >= patience:
                break
    if best_params is not None:
        model.update(best_params)
        mx.eval(model.parameters())
    return losses


def save(model: NetReturnRegressor, path: Path) -> None:
    model.save_weights(str(path))


def _hidden_sizes(weights: dict[str, mx.array]) -> tuple[int, ...]:
    """The hidden layer widths a saved model was built with, read off its
    weight shapes, so loading needs no record of how it was configured."""
    if "layer1.weight" in weights:
        return (int(weights["layer1.weight"].shape[0]),)
    count = len([name for name in weights if name.startswith("layers.") and name.endswith(".weight")])
    return tuple(int(weights[f"layers.{i}.weight"].shape[0]) for i in range(count - 1))


def load(path: Path, input_dim: int) -> NetReturnRegressor:
    model = NetReturnRegressor(input_dim=input_dim, hidden=_hidden_sizes(mx.load(str(path))))
    model.load_weights(str(path))
    return model
