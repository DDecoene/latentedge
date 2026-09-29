import json
import threading
from pathlib import Path

import numpy as np
import pytest

from latentedge.training_data import AssembledTrainingData, SplitArrays
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.train_screen import TrainScreen


def _fake_assemble(swaps_path: Path, on_progress=None) -> AssembledTrainingData:
    def split(seed: int, n: int) -> SplitArrays:
        rng = np.random.RandomState(seed)
        return SplitArrays(x=rng.randn(n, 3).astype("float32"), y=rng.randn(n).astype("float32"))

    stats = {"f0": (0.0, 1.0), "f1": (0.0, 1.0), "f2": (0.0, 1.0), "net_return": (0.0, 1.0)}
    return AssembledTrainingData(
        train=split(0, 10), validate=split(2, 4), test=split(3, 4), input_dim=3, stats=stats,
    )


def _fake_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
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

    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
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
    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
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
    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
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

    def gated_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
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


@pytest.mark.asyncio
async def test_train_screen_ctrl_q_cancels_a_running_assembly_and_exits(tmp_path: Path):
    started = threading.Event()
    unwound = threading.Event()

    def slow_assemble(swaps_path: Path, on_progress=None) -> AssembledTrainingData:
        started.set()
        try:
            for i in range(10_000):
                on_progress("labeling bars", i, 10_000)
                threading.Event().wait(0.005)
            return _fake_assemble(swaps_path)
        finally:
            unwound.set()

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet", out_path=tmp_path / "model.safetensors",
        epochs=5, assemble_fn=slow_assemble, train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert started.wait(5)
        await pilot.press("ctrl+q")
        for _ in range(100):
            await pilot.pause(0.05)
            if not app.is_running:
                break
        assert not app.is_running
        # the worker must have unwound cooperatively before the app exited,
        # not been orphaned mid-computation
        assert unwound.is_set()
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.asyncio
async def test_train_screen_shows_assembly_progress(tmp_path: Path):
    release = threading.Event()

    def gated_assemble(swaps_path: Path, on_progress=None) -> AssembledTrainingData:
        on_progress("labeling bars", 50, 100)
        release.wait(5)
        return _fake_assemble(swaps_path)

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet", out_path=tmp_path / "model.safetensors",
        epochs=2, assemble_fn=gated_assemble, train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)
    try:
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            text = str(screen.query_one("#train-progress-detail").render())
    finally:
        release.set()
    assert "labeling bars" in text


def _fake_backtest(observer):
    observer("replaying test window")
    return {
        "bars": 10, "total_return_usd": 12.5, "total_return_fraction": 0.00125,
        "max_drawdown_usd": 3.0, "win_rate": 0.6, "num_trades": 5, "sharpe": 1.1,
    }


async def _run_train_screen(tmp_path: Path, **kwargs) -> tuple[TrainScreen, LatentEdgeApp]:
    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet", out_path=tmp_path / "model.safetensors", epochs=3,
        assemble_fn=_fake_assemble, train_fn=_fake_train, **kwargs,
    )
    return screen, LatentEdgeApp(start_screen=screen)


@pytest.mark.asyncio
async def test_backtest_starts_automatically_after_training_when_the_flag_is_on(tmp_path: Path):
    from latentedge.tui.backtest_screen import BacktestScreen

    screen, app = await _run_train_screen(tmp_path, backtest_fn=_fake_backtest, backtest_after_train=True)
    async with app.run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.01)
            if isinstance(app.screen, BacktestScreen) and app.screen.is_complete:
                break
        assert isinstance(app.screen, BacktestScreen)
        assert app.screen.summary is not None
        assert app.screen.summary["num_trades"] == 5


@pytest.mark.asyncio
async def test_backtest_waits_for_b_when_the_flag_is_off(tmp_path: Path):
    from latentedge.tui.backtest_screen import BacktestScreen

    screen, app = await _run_train_screen(tmp_path, backtest_fn=_fake_backtest)
    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        assert app.screen is screen
        await pilot.press("b")
        for _ in range(100):
            await pilot.pause(0.01)
            if isinstance(app.screen, BacktestScreen) and app.screen.is_complete:
                break
        assert isinstance(app.screen, BacktestScreen)
        assert app.screen.is_complete


@pytest.mark.asyncio
async def test_b_does_nothing_before_training_completes(tmp_path: Path):
    from latentedge.tui.backtest_screen import BacktestScreen

    gate = threading.Event()

    def slow_train(model, features, labels, epochs, learning_rate, on_epoch=None, **_kwargs):
        gate.wait(5)
        return _fake_train(model, features, labels, epochs, learning_rate, on_epoch)

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet", out_path=tmp_path / "model.safetensors", epochs=3,
        assemble_fn=_fake_assemble, train_fn=slow_train, backtest_fn=_fake_backtest,
    )
    app = LatentEdgeApp(start_screen=screen)
    async with app.run_test() as pilot:
        await pilot.press("b")
        await pilot.pause(0.05)
        assert not isinstance(app.screen, BacktestScreen)
        gate.set()
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
