from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class NetReturnRegressor(nn.Module):  # type: ignore[misc]  # mlx ships no type stubs
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.layer1(x))
        result: mx.array = self.layer2(h).squeeze(-1)
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
) -> list[float]:
    x = mx.array(features)
    y = mx.array(labels)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    losses: list[float] = []
    for epoch in range(epochs):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_value = float(loss)
        losses.append(loss_value)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs, loss_value)
    return losses


def save(model: NetReturnRegressor, path: Path) -> None:
    model.save_weights(str(path))


def load(path: Path, input_dim: int) -> NetReturnRegressor:
    model = NetReturnRegressor(input_dim=input_dim)
    model.load_weights(str(path))
    return model
