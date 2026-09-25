import json
import threading
from pathlib import Path

import numpy as np
import pytest

from latentedge.training_data import AssembledTrainingData, SplitArrays
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.train_screen import TrainScreen


def _fake_assemble(swaps_path: Path) -> AssembledTrainingData:
    def split(seed: int, n: int) -> SplitArrays:
        rng = np.random.RandomState(seed)
        return SplitArrays(x=rng.randn(n, 3).astype("float32"), y=rng.randn(n).astype("float32"))

    stats = {"f0": (0.0, 1.0), "f1": (0.0, 1.0), "f2": (0.0, 1.0)}
    return AssembledTrainingData(
        train=split(0, 10), validate=split(2, 4), test=split(3, 4), input_dim=3, stats=stats,
    )


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
    assert screen.validate_correlation is not None
    assert (tmp_path / "model.safetensors").exists()

    metrics_path = tmp_path / "model.safetensors.metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["epochs"] == 5
    assert metrics["final_loss"] == pytest.approx(0.2)
    assert set(metrics["splits"]) == {"train", "validate", "test"}
    assert "correlation" in metrics["splits"]["validate"]


@pytest.mark.asyncio
async def test_train_screen_saves_stats_only_after_the_model_is_saved(tmp_path: Path):
    # Regression test: if training or saving the model fails, the stats
    # file must not exist either — a stats file with no matching model
    # (or a stale one from a previous run) would make SignalClient
    # standardize inputs with the wrong parameters at inference time.
    model_path = tmp_path / "model.safetensors"
    stats_path = tmp_path / "model.safetensors.stats.json"
    metrics_path = tmp_path / "model.safetensors.metrics.json"

    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None):
        raise RuntimeError("bad shapes")

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=model_path,
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

    assert screen.error is not None
    assert not model_path.exists()
    assert not stats_path.exists()
    assert not metrics_path.exists()


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


@pytest.mark.asyncio
async def test_train_screen_q_exits_after_completion(tmp_path: Path):
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
        action_bar_text = str(app.screen.query_one("#train-action-bar").content)
        await pilot.press("q")
        await pilot.pause()
        still_running = app.is_running

    assert "exit" in action_bar_text.lower()
    assert not still_running


@pytest.mark.asyncio
async def test_train_screen_q_exits_after_error(tmp_path: Path):
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
        action_bar_text = str(app.screen.query_one("#train-action-bar").content)
        await pilot.press("q")
        await pilot.pause()
        still_running = app.is_running

    assert "exit" in action_bar_text.lower()
    assert not still_running


@pytest.mark.asyncio
async def test_train_screen_q_ignored_before_completion(tmp_path: Path):
    release_train = threading.Event()

    def gated_train(model, features, labels, epochs, learning_rate, on_epoch=None):
        release_train.wait()
        return [1.0]

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=gated_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            still_running = app.is_running
    finally:
        release_train.set()

    assert still_running
