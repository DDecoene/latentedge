from pathlib import Path

import numpy as np
import pytest

from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.train_screen import TrainScreen


def _fake_assemble(swaps_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    features = np.random.RandomState(0).randn(10, 3).astype("float32")
    labels = np.random.RandomState(1).randn(10).astype("float32")
    return features, labels, 3


def _fake_train(model, features, labels, epochs, learning_rate, on_epoch=None):
    losses = []
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1)
        losses.append(loss)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs, loss)
    return losses


@pytest.mark.asyncio
async def test_train_screen_reaches_complete_state(tmp_path: Path):
    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.final_loss == pytest.approx(0.2)
    assert (tmp_path / "model.safetensors").exists()


@pytest.mark.asyncio
async def test_train_screen_progress_reaches_full_bar(tmp_path: Path):
    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        detail_text = str(app.screen.query_one("#train-progress-detail").content)

    assert "100%" in detail_text


@pytest.mark.asyncio
async def test_train_screen_logs_error_without_crashing(tmp_path: Path):
    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None):
        raise RuntimeError("bad shapes")

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=failing_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.error is not None:
                break

    assert "bad shapes" in screen.error
